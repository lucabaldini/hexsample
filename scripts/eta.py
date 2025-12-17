"""Analyze eta function to reconstruct events position.
"""

import numpy as np
import scipy.stats
from aptapy.hist import Histogram1d, Histogram2d, Histogram3d
from aptapy.models import PowerLaw
from aptapy.plotting import plt
from scipy.optimize import curve_fit
from tqdm import tqdm

from hexsample.app import ArgumentParser, check_required_args
from hexsample.clustering import ClusteringNN
from hexsample.fileio import DigiInputFileCircular, peek_readout_type
from hexsample.hexagon import HexagonalLayout
from hexsample.logging_ import logger
from hexsample.readout import HexagonalReadoutCircular, HexagonalReadoutMode

__description__ = \
"""Run the calibration of the eta function
"""

# Parser object.
HXETA_ARGPARSER = ArgumentParser(description=__description__)
HXETA_ARGPARSER.add_infile()

NUMBINS = 20


def mask_topology(x, y):
    """Check if the angle between the two vectors is 60 degrees (adjacent pixels)
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

def hxeta(**kwargs):
    """Application to calibrate the eta function.
    """
    check_required_args(hxeta, "infile", **kwargs)
    input_file_path = str(kwargs["infile"])
    if not input_file_path.endswith(".h5"):
        raise RuntimeError(f"Input file {input_file_path} does not look like a HDF5 file")

    readout_mode = peek_readout_type(input_file_path)
    if readout_mode is not HexagonalReadoutMode.CIRCULAR:
        raise RuntimeError("Only CIRCULAR readout is supported.")
    input_file = DigiInputFileCircular(input_file_path)
    header = input_file.header
    args = HexagonalLayout(header['layout']), header['numcolumns'], header['numrows'],\
        header['pitch'], header['noise'], header['gain']
    readout = HexagonalReadoutCircular(*args)
    nneighbors = 6
    logger.info(f'Readout chip: {readout}')
    clustering = ClusteringNN(readout, header["zsupthreshold"], nneighbors)
    # Create all the list we need to fill
    size, x0, y0, absx, absy, eta, n = [[] for _ in range(7)]
    for i, event in tqdm(enumerate(input_file)):
        cluster = clustering.run(event)
        # This cut will be made at the zero supression level
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
    # Calculate the photon position with respsect to the central pixel
    photon_pos = np.array([absx - x0, absy - y0]).T / header['pitch']

    # 2-pixel events calibration
    mask_2pix = size == 2
    eta_2pix = eta[mask_2pix].flatten()
    plt.figure("Eta distribution 2-pixel")
    eta_hist = Histogram1d(np.linspace(0, 0.5, NUMBINS), xlabel="eta")
    eta_hist.fill(eta_2pix)
    eta_hist.plot()
    photon_pos_2pix = photon_pos[mask_2pix]
    n_2pix = n[mask_2pix]
    x_binning = np.linspace(0., 0.5, NUMBINS + 1)
    # Upper limit set to 0.6 to have some margin
    y_binning = np.linspace(0., 0.6, 100)
    # We consider the projection of the photon position along the direction of the line that
    # connects the two pixels.
    dr_2pix = abs(np.sum(photon_pos_2pix * n_2pix, axis=1))
    plt.figure("2-pixel events")
    # Fill the 2D histogram
    hist = Histogram2d(x_binning, y_binning, xlabel="eta", ylabel="r / pitch")
    hist.fill(eta_2pix, dr_2pix)
    hist.plot()
    # Calculate mean and rms along r axis
    r_mean_hist, r_rms_hist = hist.collapse_axis(1)
    r_mean = r_mean_hist.content
    # We want the stddev of the mean
    r_mean_std = np.sqrt(r_rms_hist.content**2 - r_mean**2) / np.sqrt(np.sum(hist.content))
    x_fit_2pix = hist.bin_centers(axis=0)
    plt.figure("r vs eta 2-pixel")
    plt.errorbar(x_fit_2pix, r_mean, yerr=r_mean_std, fmt='.k')
    # Fit with power law
    model = PowerLaw(0.5)
    model.prefactor.freeze(0.5)
    model.fit(x_fit_2pix, r_mean, sigma=r_mean_std, absolute_sigma=True)
    model.set_plotting_range(0, 0.5)
    model.plot(fit_output=True)
    plt.xlabel("eta")
    plt.ylabel("r / pitch")
    plt.xscale('linear')
    plt.yscale('linear')

    def probit(x, a):
        """Probit function to fit theta vs eta ratio.
        """
        f = scipy.stats.norm(0, 0.14)
        return a*f.ppf(x) + 0.5

    popt, pcov = curve_fit(probit, x_fit_2pix, r_mean, sigma=r_mean_std, absolute_sigma=True)
    xx = np.linspace(0, 0.5, 100)
    chisq = np.sum(((r_mean - probit(x_fit_2pix, *popt)) / r_mean_std)**2)
    plt.plot(xx, probit(xx, *popt), 'r--', label=f"Probit fit\nchisq / ndof: {chisq:.2f} / \
             {(len(r_mean) - 1)}\na = {popt[0]:.3f} +/- {np.sqrt(pcov[0,0]):.3f}")
    plt.legend()

    # 3-pixel events calibration
    mask_3pix = size == 3
    eta_3pix = np.stack(eta[mask_3pix])
    photon_pos_3pix = photon_pos[mask_3pix]
    n_3pix = n[mask_3pix]
    r = np.sqrt(np.sum(photon_pos_3pix**2, axis=1))
    cos_theta = np.sum(photon_pos_3pix * n_3pix, axis=1) / r
    theta = np.arccos(cos_theta)
    plt.figure("Theta distribution 3-pixel")
    theta_hist = Histogram1d(np.linspace(0, np.pi, 100), xlabel="theta [rad]")
    theta_hist.fill(theta)
    theta_hist.plot()

    eta1, eta2 = eta_3pix[:, 0], eta_3pix[:, 1]
    eta_sum = eta1 + eta2
    x_r_binning = np.linspace(0., 2/3, NUMBINS + 1)
    x_theta_binning = np.linspace(0.2, 2/3, 2)
    eta_diff = (eta1 - eta2) / (eta1 + eta2)
    y_binning = np.linspace(0., 1, NUMBINS + 1)
    y_r_binning = np.linspace(0., 1, 2)

    xlabel = "eta1 + eta2"
    ylabel = "(eta1 - eta2) / (eta1 + eta2)"
    r_binning = np.linspace(0., 1/np.sqrt(3), 101)
    # This is just to plot, we don't need to analyze this histogram
    plt.figure("r vs eta sum and diff 3-pixel")
    r_hist3d = Histogram3d(x_r_binning, y_binning, r_binning, xlabel=xlabel, ylabel=ylabel,
                           zlabel="r")
    r_hist3d.fill(eta_sum, eta_diff, r)
    r_mean, _ = r_hist3d.collapse_axis(2)
    r_mean.zlabel = "r mean"
    r_mean.plot()

    # Create a histogram slice to fit r vs eta sum
    r_slice = Histogram3d(x_r_binning, y_r_binning, r_binning, xlabel=xlabel, ylabel=ylabel,
                          zlabel="r")
    r_slice.fill(eta_sum, eta_diff, r)
    r_slice_mean, r_slice_rms = r_slice.collapse_axis(2)
    # Plot r vs eta sum
    plt.figure("r vs eta sum - 3 pixels")
    x_fit = r_slice_mean.bin_centers(axis=0)
    y_fit = r_slice_mean.content.flatten()
    y_err = np.sqrt(r_slice_rms.content.flatten()**2 - y_fit**2) / np.sqrt(np.sum(r_slice.content))
    plt.errorbar(x_fit, y_fit, yerr=y_err, fmt=".k")
    plt.xlabel("eta1 + eta2")
    plt.ylabel("r / pitch")
    # Fit with power law
    model = PowerLaw(2/3)
    model.prefactor.freeze(1 / np.sqrt(3))
    model.fit(x_fit, y_fit, sigma=y_err, absolute_sigma=True)
    model.set_plotting_range(0, 2/3)
    model.plot(fit_output=True)
    plt.xscale('linear')
    plt.yscale('linear')
    plt.legend()
    # Calibration of the angle theta vs eta diff ratio
    # This histogram is just to plot
    theta_binning = np.linspace(0, np.pi/6, 101)
    theta_hist3d = Histogram3d(x_r_binning, y_binning, theta_binning, xlabel=xlabel, ylabel=ylabel,
                               zlabel="theta")
    theta_hist3d.fill(eta_sum, eta_diff, theta)
    theta_mean, _ = theta_hist3d.collapse_axis(2)
    plt.figure("theta mean")
    theta_mean.plot()
    # Create histogram slice to fit theta vs eta diff ratio
    theta_slice = Histogram3d(x_theta_binning, y_binning, theta_binning, xlabel=xlabel,
                              ylabel=ylabel, zlabel="theta")
    theta_slice.fill(eta_sum, eta_diff, theta)
    theta_slice_mean, theta_slice_rms = theta_slice.collapse_axis(2)
    mask_zero = theta_slice_mean.content.flatten() > 0
    x_fit = theta_slice_mean.bin_centers(axis=1)[mask_zero]
    y_fit = theta_slice_mean.content.flatten()[mask_zero]
    N = np.sum(theta_slice.content)
    # Calculate the standard deviation of the mean
    y_err = np.sqrt(theta_slice_rms.content.flatten()[mask_zero]**2 - y_fit**2) / np.sqrt(N)
    # This is just to make a comparison, a power law is not good
    plt.figure("theta vs eta diff ratio")
    model = PowerLaw()
    model.prefactor.freeze(np.pi/6)
    model.fit(x_fit, y_fit, sigma=y_err, absolute_sigma=True)
    model.plot(fit_output=True)
    # Fitting with a better model
    def fit_func(x, a):
        return (np.pi/6) * (np.exp(x*a) - 1) / (np.exp(a) - 1)
    popt, pcov = curve_fit(fit_func, x_fit, y_fit, sigma=y_err, absolute_sigma=True)
    xx = np.linspace(0, 1, 100)
    print(f"Fit parameter: gamma = {popt[0]} +/- {np.sqrt(pcov[0,0])}")
    plt.plot(xx, fit_func(xx, *popt), 'r--')
    plt.errorbar(x_fit, y_fit, yerr=y_err, fmt=".k")
    plt.xlabel("(eta1 - eta2) / (eta1 + eta2)")
    plt.ylabel("theta [rad]")
    plt.xscale("linear")
    plt.yscale("linear")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    hxeta(**vars(HXETA_ARGPARSER.parse_args()))