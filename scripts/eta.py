"""Analyze eta function to reconstruct events position.
"""

from hexsample.cli import argparse

import arviz as az
import numpy as np
from aptapy.hist import Histogram2d
from aptapy.models import Probit, Exponential, Constant, ExponentialComplement, StretchedExponentialComplement
from aptapy.plotting import plt
from tqdm import tqdm

from hexsample.clustering import ClusteringNN
from hexsample.fileio import digi_input_file_class, peek_readout_type
from hexsample.hexagon import HexagonalLayout
from hexsample.logging_ import logger
from hexsample.readout import HexagonalReadoutCircular, HexagonalReadoutMode

__description__ = \
"""Run the calibration of the eta function.
"""

# Parser object.
HXETA_ARGPARSER = argparse.ArgumentParser(description=__description__)
HXETA_ARGPARSER.add_argument("input_file", type=str, help="path to the input file")

NUMBINS = 20
STATISTIC = "median"

def mask_topology(x, y):
    """Check whether the pixel topology corresponds to adjacent pixels, by checking the angle
    between the vectors that connect the pixel with the highest pha to the other pixels.

    This function will not be useful when the zero suppression will be improved to remove noisy
    pixels from events.
    """
    if x.shape[1] == 2:
        mask = np.full(x.shape[0], True)
    elif x.shape[1] == 3:
        u = np.array([x[:, 1] - x[:, 0], y[:, 1] - y[:, 0]]).T
        v = np.array([x[:, 2] - x[:, 0], y[:, 2] - y[:, 0]]).T
        cos_theta = np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        mask = np.isclose(cos_theta, 0.5, atol=1e-2)
    else:
        mask = np.full(x.shape[0], False)
    return mask


def _estimate_loc(hist: Histogram2d, statistic: str, debug: bool = False):
    """Estimate the location of the distribution in each slice of the histogram.
    """
    centers = hist.bin_centers(axis=0)
    loc = np.zeros(centers.shape)
    err = np.zeros(centers.shape)
    for i in range(len(centers)):
        slice_ = hist.slice1d(i)
        n = slice_.content.sum()
        if n == 0:
            loc[i] = np.nan
            err[i] = np.nan
            continue
        # For each eta bin, extract the slice (dr distribution) and estimate the location as the
        # center of the minimum interval containing 68% of the distribution. The uncertainty is
        # estimated as half the width of this interval divided by sqrt(N).
        if statistic == "hdi":
            x = np.repeat(slice_.bin_centers(), slice_.content.astype(int))
            xmin, xmax = az.hdi(x, hdi_prob=0.6827)
            loc[i] = (xmax + xmin) / 2
            err[i] = (xmax - xmin) / (2 * n**0.5)
        # Estimate the location as the mean
        elif statistic == "mean":
            mean, std = slice_.binned_statistics()
            loc[i] = mean
            err[i] = std / np.sqrt(n)
        # Estimate the location as the median
        elif statistic == "median":
            x = np.repeat(slice_.bin_centers(), slice_.content.astype(int))
            loc[i] = np.median(x)
            err[i] = 1.253 * np.std(x) / np.sqrt(n)  # Approximate std error of median
        # Plot for debugging
        if debug:
            plt.figure(f"debug_slices_{i}")
            slice_.plot()
            plt.vlines(loc[i], 0, max(slice_.content), color="r")
    # Remove NaN entries
    mask = ~np.isnan(loc)
    centers = centers[mask]
    loc = loc[mask]
    err = err[mask]
    return centers, loc, err


def calibrate_2pix(eta, photon_pos, versor, statistic):
    # Define binning and calculate dr, which is the distance of the photon from the center of the
    # most charged pixel, projected onto the line connecting the two pixels
    eta_binning = np.linspace(0., 0.5, NUMBINS + 1)
    dr_binning = np.linspace(0., 0.6, 101)
    dr = abs(np.sum(photon_pos * versor, axis=1))
    # Create the histogram of dr vs eta and fill it
    plt.figure("dr_vs_eta_2pix")
    hist = Histogram2d(eta_binning, dr_binning, xlabel="eta", ylabel="dr / p")
    hist.fill(eta, dr)
    hist.plot()
    # Now the calibration. We analyze each column of the histogram and calculate a statistic
    # to estimate the location of the distribution and its uncertainty.
    eta_centers, dr_loc, dr_err = _estimate_loc(hist, statistic, debug=False)
    # Now fit the data with a probit model and plot the results
    model = Probit()
    model.offset.freeze(0.5)
    model.fit(eta_centers, dr_loc, sigma=dr_err, absolute_sigma=True)
    plt.figure("dr_vs_eta_2pix_calibration")
    plt.errorbar(eta_centers, dr_loc, yerr=dr_err, fmt=".k", label="Data")
    model.plot(fit_output=True)
    plt.xlabel("eta")
    plt.ylabel("dr / p")
    plt.legend()

    return model


def calibrate_dr_3pix(eta, photon_pos, statistic: str):
    # Calculate dr, which is the distance of the photon from the center of the
    # most charged pixel.
    dr = np.sqrt(np.sum(photon_pos**2, axis=1))
    # Calculate eta+ and define the binning
    eta_sum = np.sum(eta, axis=1)
    eta_binning = np.linspace(0., 2/3, NUMBINS + 1)
    dr_binning = np.linspace(0., 1 / np.sqrt(3), 101)
    # Create the histogram of dr vs eta+ and fill it
    plt.figure("dr_vs_eta_sum_3pix") 
    hist = Histogram2d(eta_binning, dr_binning, xlabel="eta1 + eta2", ylabel="dr / p")
    hist.fill(eta_sum, dr)
    hist.plot()
    # Now the calibration.
    eta_centers, dr_loc, dr_err = _estimate_loc(hist, statistic, debug=False)
    # Fit with a probit model and plot the results
    model = Probit()
    model.fit(eta_centers, dr_loc, sigma=dr_err, absolute_sigma=True)
    plt.figure("dr_vs_eta_sum_3pix_calibration")
    plt.errorbar(eta_centers, dr_loc, yerr=dr_err, fmt=".k", label="Data")
    model.set_plotting_range(0, model.plotting_range()[1])
    model.plot(fit_output=True)
    plt.xlabel("eta1 + eta2")
    plt.ylabel("dr / p")
    plt.legend()

    return model


def calibrate_theta_3pix(eta, photon_pos, versor, statistic: str):
    # Calculate theta
    r = np.sqrt(np.sum(photon_pos**2, axis=1))
    cos_theta = np.sum(photon_pos * versor, axis=1) / r
    theta = np.arccos(cos_theta)
    theta_binning = np.linspace(0., max(theta), 100)
    # Calculate eta- and define the binning
    eta_sum = np.sum(eta, axis=1)
    eta_diff = (eta[:, 0] - eta[:, 1]) / eta_sum
    eta_binning = np.linspace(0., 1., NUMBINS + 1)
    # Create the histogram of theta vs eta- and fill it
    plt.figure("theta_vs_eta_diff_3pix") 
    hist = Histogram2d(eta_binning, theta_binning, xlabel="(eta1 - eta2) / (eta1 + eta2)",
                       ylabel="theta [rad]")
    hist.fill(eta_diff, theta)
    hist.plot()
    # Now the calibration.
    eta_centers, theta_loc, theta_err = _estimate_loc(hist, statistic, debug=False)
    # Fit with an exponential model and plot the results
    model = Exponential() 
    model.fit(eta_centers, theta_loc, sigma=theta_err, absolute_sigma=True)
    plt.figure("theta_vs_eta_diff_3pix_calibration")
    plt.errorbar(eta_centers, theta_loc, yerr=theta_err, fmt=".k", label="Data")
    model.set_plotting_range(0, model.plotting_range()[1])
    model.plot(fit_output=True)
    plt.xlabel("(eta1 - eta2) / (eta1 + eta2)")
    plt.ylabel("theta [rad]")
    plt.legend()

    return model


def hxeta(**kwargs):
    """Application to calibrate the eta function.
    """
    input_file_path = str(kwargs["input_file"])
    if not input_file_path.endswith(".h5"):
        raise RuntimeError(f"Input file {input_file_path} does not look like a HDF5 file")

    readout_mode = peek_readout_type(input_file_path)
    if readout_mode is not HexagonalReadoutMode.CIRCULAR:
        raise RuntimeError("Only CIRCULAR readout is supported.")
    file_type = digi_input_file_class(readout_mode)
    input_file = file_type(input_file_path)
    header = input_file.header
    header["zero_sup_threshold"] = 0
    args = HexagonalLayout(header["layout"]), header["num_cols"], header["num_rows"],\
        header["pitch"], header["enc"], header["gain"], header["zero_sup_threshold"]
    readout = HexagonalReadoutCircular(*args)
    nneighbors = 6
    logger.info(f"Readout chip: {readout}")
    clustering = ClusteringNN(readout, header["zero_sup_threshold"], nneighbors)
    # Create all the lists we need to fill
    size, x0, y0, absx, absy, eta, n = [[] for _ in range(7)]
    for i, event in tqdm(enumerate(input_file)):
        cluster = clustering.run(event)
        # This cut will be made at the zero suppression level
        valid_topology = mask_topology(cluster.x.reshape(1, -1), cluster.y.reshape(1, -1))[0]
        if (cluster.size() == 2 or cluster.size() == 3) and valid_topology:
            # Check if the event topology is acceptable
            size.append(cluster.size())
            x0.append(cluster.x[0])
            y0.append(cluster.y[0])
            eta.append(cluster.calculate_eta())
            n.append(cluster.n_versor())
            mc_event = input_file.mc_event(i)
            absx.append(mc_event.absx)
            absy.append(mc_event.absy)
    input_file.close()

    size = np.array(size)
    x0 = np.array(x0)
    y0 = np.array(y0)
    absx = np.array(absx)
    absy = np.array(absy)
    eta = np.array(eta, dtype=object)
    n = np.array(n)
    # Calculate the photon position with respect to the central pixel
    photon_pos = np.array([absx - x0, absy - y0]).T / header["pitch"]

    # Select two pixel events and calibrate
    mask_2pix = size == 2
    eta_2pix = eta[mask_2pix].flatten()
    photon_pos_2pix = photon_pos[mask_2pix]
    n_2pix = n[mask_2pix]
    model_2pix = calibrate_2pix(eta_2pix, photon_pos_2pix, n_2pix, statistic=STATISTIC)

    # Select three pixel events and calibrate
    mask_3pix = size == 3
    eta_3pix = np.stack(eta[mask_3pix])
    photon_pos_3pix = photon_pos[mask_3pix]
    n_3pix = n[mask_3pix]
    model_3pix_r = calibrate_dr_3pix(eta_3pix, photon_pos_3pix, statistic=STATISTIC)
    model_3pix_theta = calibrate_theta_3pix(eta_3pix, photon_pos_3pix, n_3pix, statistic=STATISTIC)
    
    return model_2pix, model_3pix_r, model_3pix_theta

if __name__ == "__main__":
    hxeta(**vars(HXETA_ARGPARSER.parse_args()))
