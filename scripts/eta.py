"""Analyze eta function to reconstruct events position.
"""

from pathlib import Path

import numpy as np
from aptapy.hist import Histogram1d, Histogram2d
from aptapy.modeling import AbstractFitModel
from aptapy.models import Probit
from aptapy.plotting import last_line_color, plt
from tqdm import tqdm

from hexsample.cli import argparse
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
HXETA_ARGPARSER.add_argument("--save", action="store_true",
                            help="save the calibration plots to the results directory")


NUMBINS = 20
RESULTS_DIR = Path.home() / "hexsample_figures"
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def mask_topology(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Check whether the pixel topology corresponds to adjacent pixels, by checking the angle
    between the vectors that connect the pixel with the highest pha to the other pixels.

    This function will not be useful when the zero suppression will be improved to remove noisy
    pixels from events.

    Arguments
    ---------
    x : np.ndarray
        The x positions of the pixels in the cluster.
    y : np.ndarray
        The y positions of the pixels in the cluster.
    
    Returns
    -------
    mask : np.ndarray
        A boolean mask indicating whether each cluster has a valid topology.
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


def _estimate_loc(hist: Histogram2d, debug: bool = False
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the location of the distribution in each slice of the histogram.

    Arguments
    ---------
    hist : Histogram2d
        The 2D histogram to analyze.
    debug : bool, optional
        If True, plot each slice with the estimated location.
    
    Returns
    -------
    centers : np.ndarray
        The bin centers of the first axis.
    loc : np.ndarray
        The estimated location of the distribution in each slice.
    err : np.ndarray
        The estimated uncertainty on the location in each slice.
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
        # Estimate the location as the median of the distribution and its error
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


def calibrate_2pix(eta: np.ndarray, photon_pos: np.ndarray, versors: np.ndarray,
                   **kwargs) -> AbstractFitModel:
    """Calibrate the 2-pixel eta function.
    
    Arguments
    ---------
    eta : np.ndarray
        The eta values for the 2-pixel clusters.
    photon_pos : np.ndarray
        The photon positions with respect to the central pixel, in units of pitch.
    versors : np.ndarray
        The versors defined for the 2-pixel clusters.
    
    Returns
    -------
    model : AbstractFitModel
        The calibrated probit model.
    """
    # Define binning and calculate dr, which is the distance of the photon from the center of the
    # most charged pixel, projected onto the line connecting the two pixels
    eta_binning = np.linspace(0., 0.5, NUMBINS + 1)
    dr_binning = np.linspace(0., 0.6, 101)
    dr = abs(np.sum(photon_pos * versors[:, 0], axis=1))
    # Create the histogram of dr vs eta and fill it
    plt.figure("dr_vs_eta_2pix")
    hist = Histogram2d(eta_binning, dr_binning, xlabel=r"$\eta$", ylabel=r"r / p")
    hist.fill(eta, dr)
    hist.plot()
    # Now the calibration. We analyze each column of the histogram and calculate a statistic
    # to estimate the location of the distribution and its uncertainty.
    eta_centers, dr_loc, dr_err = _estimate_loc(hist, debug=False)
    # Now fit the data with a probit model and plot the results
    model = Probit()
    model.offset.freeze(0.5)
    model.fit(eta_centers, dr_loc, sigma=dr_err, absolute_sigma=True)
    fig = plt.figure("dr_vs_eta_2pix_calibration")
    plt.errorbar(eta_centers, dr_loc, yerr=dr_err, fmt=".k", label="Monte Carlo simulation")
    fit_label = "2-pixel events calibration\n" + fr"$\sigma$ = {model.sigma.ufloat()}"
    model.plot(label=fit_label, color=last_line_color())
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"r / p")
    plt.legend()
    if kwargs.get("save", False):
        fig.savefig(RESULTS_DIR / "2pix_cal.pdf", format="pdf")

    eta_min = min(eta)[0]
    eta_hist_binning = np.linspace(eta_min, 0.5, 101)
    eta_hist = Histogram1d(eta_hist_binning, xlabel="eta", ylabel="Counts")
    eta_hist.fill(eta)
    plt.figure("eta_2pix_distribution")
    eta_hist.plot()

    from scipy.special import ndtri

    sigma_x = model.sigma.value * np.sqrt(2 * np.pi) * np.exp(0.5 * abs(ndtri(eta_binning))**2) * np.sqrt(eta_binning**2  + (1 - eta_binning)**2) * 30 / 1600
    plt.figure("dr_vs_eta_2pix_derivative")
    plt.plot(model(eta_binning), sigma_x, label="d(dr/p)/d(eta)")

    plt.figure("dr_distr")
    plt.hist(model(eta.astype(float)), bins=100, label="dr / p")
    plt.legend()

    return model


def calibrate_dr_3pix(eta: np.ndarray, photon_pos: np.ndarray, **kwargs) -> AbstractFitModel:
    """Calibrate the dr component of the 3-pixel eta function.
    
    Arguments
    ---------
    eta : np.ndarray
        The eta values for the 3-pixel clusters.
    photon_pos : np.ndarray
        The photon positions with respect to the central pixel, in units of pitch.
    
    Returns
    -------
    model : AbstractFitModel
        The calibrated probit model.
    """
    # Calculate dr, which is the distance of the photon from the center of the
    # most charged pixel.
    dr = np.sqrt(np.sum(photon_pos**2, axis=1))
    # Calculate eta+ and define the binning
    eta_sum = np.sum(eta, axis=1)
    eta_binning = np.linspace(0., 2/3, NUMBINS + 1)
    dr_binning = np.linspace(0., 1 / np.sqrt(3), 101)
    # Create the histogram of dr vs eta+ and fill it
    plt.figure("dr_vs_eta_sum_3pix")
    hist = Histogram2d(eta_binning, dr_binning, xlabel=r"$\eta^+$", ylabel=r"r / p")
    hist.fill(eta_sum, dr)
    hist.plot()
    # Now the calibration.
    eta_centers, dr_loc, dr_err = _estimate_loc(hist, debug=False)
    # Fit with a probit model and plot the results
    model = Probit()
    model.fit(eta_centers, dr_loc, sigma=dr_err, absolute_sigma=True)
    fig = plt.figure("dr_vs_eta_sum_3pix_calibration")
    plt.errorbar(eta_centers, dr_loc, yerr=dr_err, fmt=".k", label="Monte Carlo simulation")
    model.set_plotting_range(0, model.plotting_range()[1])
    fit_label = "3-pixel events radial calibration\n" + fr"$\sigma$ = {model.sigma.ufloat()}"
    fit_label += "\n" + fr"$\mu$ = {model.offset.ufloat()}"
    model.plot(label=fit_label, color=last_line_color())
    plt.xlabel(r"$\eta^+$")
    plt.ylabel(r"r / p")
    plt.legend()
    if kwargs.get("save", False):
        fig.savefig(RESULTS_DIR / "3pix_cal_radial.pdf", format="pdf")

    return model


def calibrate_theta_3pix(eta: np.ndarray, photon_pos: np.ndarray, versors: np.ndarray,
                         **kwargs) -> AbstractFitModel:
    """Calibrate the theta component of the 3-pixel eta function.

    Arguments
    ---------
    eta : np.ndarray
        The eta values for the 3-pixel clusters.
    photon_pos : np.ndarray
        The photon positions with respect to the central pixel, in units of pitch.
    versors : np.ndarray
        The versors defined for the 3-pixel clusters.

    Returns
    -------
    model : AbstractFitModel
        The calibrated exponential model.
    """
    dr = np.sqrt(np.sum(photon_pos**2, axis=1))
    # Calculate theta
    u = versors[:, 0]
    v = versors[:, 1]
    # Calculate the projections onto the versors
    u_proj = np.sum(photon_pos * u, axis=1)
    v_proj = np.sum(photon_pos * v, axis=1)
    # Calculate theta as arctan(v_proj / u_proj)
    theta = np.arctan2(v_proj, u_proj)
    # Calculate the transverse component (for small angle approximation)
    y = dr * theta

    y_binning = np.linspace(min(y), max(y), 1000)
    # Calculate eta- and define the binning
    eta_sum = np.sum(eta, axis=1)
    eta_diff = (eta[:, 0] - eta[:, 1]) / eta_sum
    eta_binning = np.linspace(0., 1., NUMBINS + 1)
    # Create the histogram of theta vs eta- and fill it
    plt.figure("theta_vs_eta_diff_3pix")
    hist = Histogram2d(eta_binning, y_binning, xlabel=r"$\eta^-$",
                       ylabel=r"r$\theta$ / p")
    hist.fill(eta_diff, y)
    hist.plot()
    # Now the calibration.
    eta_centers, y_loc, y_err = _estimate_loc(hist, debug=False)
    # Fit with the Probit and plot the results
    model = Probit()
    model.offset.freeze(0.)
    model.fit((1 + eta_centers)/2, y_loc, sigma=y_err, absolute_sigma=True)
    fig = plt.figure("theta_vs_eta_diff_3pix_calibration")
    plt.errorbar(eta_centers, y_loc, yerr=y_err, fmt=".k", label="Monte Carlo simulation")
    fit_label = "3-pixel events angular calibration\n"
    fit_label += fr"$\sigma$ = {model.sigma.ufloat()}"
    xx = np.linspace(0.5, max((eta_centers + 1)/2), 100)
    plt.plot(2*xx - 1, model(xx), label=fit_label, color=last_line_color())
    plt.xlabel(r"$\eta^-$")
    plt.ylabel(r"r$\theta$ / p")
    plt.legend()
    if kwargs.get("save", False):
        fig.savefig(RESULTS_DIR / "3pix_cal_angular.pdf", format="pdf")

    return model


def hxeta(**kwargs) -> tuple[AbstractFitModel, AbstractFitModel, AbstractFitModel]:
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
    args = HexagonalLayout(header["layout"]), header["num_cols"], header["num_rows"],\
        header["pitch"], header["enc"], header["gain"], header["zero_sup_threshold"]
    readout = HexagonalReadoutCircular(*args)
    nneighbors = 6
    logger.info(f"Readout chip: {readout}")
    clustering = ClusteringNN(readout, header["zero_sup_threshold"], nneighbors)
    # Create all the lists we need to fill
    size, x0, y0, absx, absy, eta, versors = [[] for _ in range(7)]
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
            versors.append(cluster.versors())
            mc_event = input_file.mc_event(i)
            absx.append(mc_event.absx)
            absy.append(mc_event.absy)
    input_file.close()

    size = np.array(size)
    x0 = np.array(x0)
    y0 = np.array(y0)
    absx = np.array(absx)
    absy = np.array(absy)
    # Eta must be an array of objects because clusters can have different sizes
    eta = np.array(eta, dtype=object)
    versors = np.array(versors)
    # Calculate the photon position with respect to the central pixel
    photon_pos = np.array([absx - x0, absy - y0]).T / header["pitch"]

    # Select two pixel events and calibrate
    mask_2pix = size == 2
    eta_2pix = eta[mask_2pix].flatten()
    photon_pos_2pix = photon_pos[mask_2pix]
    versors_2pix = versors[mask_2pix]
    model_2pix = calibrate_2pix(eta_2pix, photon_pos_2pix, versors_2pix, **kwargs)

    # Select three pixel events and calibrate
    mask_3pix = size == 3
    eta_3pix = np.stack(eta[mask_3pix])
    photon_pos_3pix = photon_pos[mask_3pix]
    versors_3pix = versors[mask_3pix]
    model_3pix_r = calibrate_dr_3pix(eta_3pix, photon_pos_3pix, **kwargs)
    model_3pix_theta = calibrate_theta_3pix(eta_3pix, photon_pos_3pix, versors_3pix, **kwargs)

    return model_2pix, model_3pix_r, model_3pix_theta

if __name__ == "__main__":
    hxeta(**vars(HXETA_ARGPARSER.parse_args()))
    plt.show()
