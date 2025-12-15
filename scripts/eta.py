"""Analyze eta function and fit with power law
"""

from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

from aptapy.hist import Histogram3d, Histogram2d
from aptapy.models import PowerLaw
from aptapy.plotting import plt
from matplotlib.colors import LogNorm

from hexsample import HEXSAMPLE_DATA
from hexsample.clustering import ClusteringNN
from hexsample.fileio import DigiInputFileCircular
from hexsample.readout import HexagonalReadoutCircular
from hexsample.hexagon import HexagonalLayout



def calculate_eta(pha):
    if pha.shape[1] == 2:
        return pha[:, 1] / pha.sum(axis=1)
    elif pha.shape[1] == 3:
        return pha[:, 1] / pha.sum(axis=1), pha[:, 2] / pha.sum(axis=1)
    else:
        raise ValueError("Unsupported cluster size for eta calculation")


def calculate_n(x, y):
    if x.shape[1] == 2:
        u = np.array([x[:, 1] - x[:, 0], y[:, 1] - y[:, 0]]).T
        return u / np.sqrt(np.sum(u**2, axis=1, keepdims=True))
    elif x.shape[1] == 3:
        u = np.array([x[:, 1] - x[:, 0], y[:, 1] - y[:, 0]]).T
        v = np.array([x[:, 2] - x[:, 0], y[:, 2] - y[:, 0]]).T
        return u / np.sqrt(np.sum(u**2, axis=1, keepdims=True)), v / np.sqrt(np.sum(v**2, axis=1, keepdims=True))
    else:
        raise ValueError("Unsupported cluster size for n calculation")

def calculate_versor(x, y):
    if x.shape[1] == 2:
        n = np.array([x[:, 1] - x[:, 0], y[:, 1] - y[:, 0]]).T
    elif x.shape[1] == 3:
        n = np.array([x[:, 1] + x[:, 2] - 2 * x[:, 0], y[:, 1] + y[:, 2] - 2 * y[:, 0]]).T
    # We have a problem: if the two pixels are not adjacent this reconstruction fails.
    # In particular, if the two pixels are opposite, the versor is zero. There is also a second
    # disposition that gives a wrong versor (but not zero). We should filter this cases before, at
    # least for this type of reconstruction. We make the cut while reading the file.
    return n / np.sqrt(np.sum(n**2, axis=1, keepdims=True))

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
    return mask

def hxeta(npixels = 2, numbins= 10):
    input_file_path = HEXSAMPLE_DATA / "hxsim_large.h5"
    input_file = DigiInputFileCircular(input_file_path)
    header = input_file.header
    args = HexagonalLayout(header['layout']), header['numcolumns'], header['numrows'],\
        header['pitch'], header['noise'], header['gain']
    readout = HexagonalReadoutCircular(*args)
    zsupthreshold = 30
    nneighbors = 6
    logger.info(f'Readout chip: {readout}')
    clustering = ClusteringNN(readout, zsupthreshold, nneighbors)

    pha = []
    x = []
    y = []
    absx = []
    absy = []
    for i, event in tqdm(enumerate(input_file)):
        cluster = clustering.run(event)
        if cluster.size() == npixels:
            pha.append(cluster.pha)
            x.append(cluster.x)
            y.append(cluster.y)

            mc_event = input_file.mc_event(i)
            absx.append(mc_event.absx)
            absy.append(mc_event.absy)
    input_file.close()
    x = np.array(x)
    y = np.array(y)
    # See note in calculate_versor
    mask = mask_topology(x, y)
    x = x[mask]
    y = y[mask]
    pha = np.array(pha)[mask]
    absx = np.array(absx)[mask]
    absy = np.array(absy)[mask]

    n = calculate_versor(x, y)
    photon_pos = np.array([absx - x[:, 0], absy - y[:, 0]]).T / header['pitch']
    r = np.sqrt(np.sum(photon_pos**2, axis=1))
    if npixels == 2:
        x_binning = np.linspace(0., 0.5, numbins + 1)
        y_binning = np.linspace(0., 0.6, 100)
        eta = calculate_eta(pha)
        dr = abs(np.sum(photon_pos * n, axis=1))

        hist = Histogram2d(x_binning, y_binning, xlabel="eta", ylabel="r")
        hist.fill(eta, dr)
        plt.figure("r vs eta")
        hist.plot()
        r_mean = np.sum(hist.content * hist.bin_centers(axis=1)[np.newaxis, :], axis=1) / np.sum(hist.content, axis=1)
        r_std = np.sqrt(np.sum(hist.content * (hist.bin_centers(axis=1)[np.newaxis, :] - r_mean[:, np.newaxis])**2, axis=1) / np.sum(hist.content, axis=1))
        plt.figure("r mean")
        plt.errorbar(hist.bin_centers(axis=0), r_mean, yerr=r_std, fmt='.k')
        plt.xlabel("eta")
        plt.ylabel("r")
        # Fit with power law
        popt, pcov = curve_fit(lambda x, b: 1/2* (x*2)**b, hist.bin_centers(axis=0)[~np.isnan(r_mean)], r_mean[~np.isnan(r_mean)], sigma=r_std[~np.isnan(r_mean)], absolute_sigma=True)
        xx = np.linspace(0, 0.5, 100)
        plt.plot(xx, 1/2 * (xx*2)**popt[0], 'r--', label=f"fit")
        plt.xscale('linear')
        plt.yscale('linear')
        print(f"Fit parameters: b = {popt[0]:.2f} +/- {np.sqrt(pcov[0,0]):.2f}")
        plt.legend() 


    if npixels == 3:
        cos_theta = np.sum(photon_pos * n, axis=1) / r
        theta = np.arccos(cos_theta)
        plt.figure("theta")
        plt.hist(theta, bins=100)

        eta1, eta2 = calculate_eta(pha)
        x = eta1 + eta2
        x_r_binning = np.linspace(0., 2/3, numbins + 1)
        x_theta_binning = np.linspace(0.2, 2/3, 2)
        y = (eta1 - eta2) / (eta1 + eta2)
        y_binning = np.linspace(0., 1, numbins + 1)
        y_r_binning = np.linspace(0., 1, 2)

        xlabel = "eta1 + eta2"
        ylabel = "(eta1 - eta2) / (eta1 + eta2)"
        r_binning = np.linspace(0., 1/np.sqrt(3), 101)
        r_hist3d = Histogram3d(x_r_binning, y_binning, r_binning, xlabel=xlabel, ylabel=ylabel, zlabel="r")
        r_hist3d.fill(x, y, r)
        r_mean, r_rms = r_hist3d.collapse_axis(2)
        plt.figure("r mean")
        r_mean.plot()

        plt.figure("r slices")
        for i in range(numbins):
            r_slice = r_mean.content[:, i][r_mean.content[:, i] > 0]
            rms_slice = r_rms.content[:, i][r_mean.content[:, i] > 0]
            plt.plot(r_mean.bin_centers(axis=0)[r_mean.content[:, i] > 0], r_slice, label=f'x bin {i}')

        r_slice = Histogram3d(x_r_binning, y_r_binning, r_binning, xlabel=xlabel, ylabel=ylabel, zlabel="r")
        r_slice.fill(x, y, r)
        r_slice_mean, r_slice_rms = r_slice.collapse_axis(2)
        x_fit = r_slice_mean.bin_centers(axis=0)
        y_fit = r_slice_mean.content.flatten()
        y_err = np.sqrt(r_slice_rms.content.flatten()**2 - y_fit**2)
        plt.figure("r vs eta sum")
        plt.errorbar(x_fit, y_fit, yerr=y_err, fmt=".k")
        plt.xlabel("eta1 + eta2")
        plt.ylabel("r")
        # Not able to fit with curve_fit directly, need to define a function
        model = PowerLaw()
        # model.prefactor.freeze(1/np.sqrt(3) * (3/2)**model.index)
        model.fit(x_fit, y_fit, sigma=y_err, absolute_sigma=True)
        model.plot(fit_output=True)
        popt, pcov = curve_fit(lambda x, b: 1/np.sqrt(3) * (x*3/2)**b, x_fit, y_fit, sigma=y_err, absolute_sigma=True)
        xx = np.linspace(0, 2/3, 100)
        plt.plot(xx, 1/np.sqrt(3) * (xx*3/2)**popt[0], 'r--', label=f'fit: b={popt[0]:.2f}')
        plt.xscale('linear')
        plt.yscale('linear')
        plt.legend()

        theta_binning = np.linspace(0, np.pi/6, 101)
        theta_hist3d = Histogram3d(x_r_binning, y_binning, theta_binning, xlabel=xlabel, ylabel=ylabel, zlabel="theta")
        theta_hist3d.fill(x, y, theta)
        theta_mean, theta_rms = theta_hist3d.collapse_axis(2)
        plt.figure("theta mean")
        theta_mean.plot()

        theta_slice = Histogram3d(x_theta_binning, y_binning, theta_binning, xlabel=xlabel, ylabel=ylabel, zlabel="theta")
        theta_slice.fill(x, y, theta)
        theta_slice_mean, theta_slice_rms = theta_slice.collapse_axis(2)
        mask_zero = theta_slice_mean.content.flatten() > 0
        x_fit = theta_slice_mean.bin_centers(axis=1)[mask_zero]
        y_fit = theta_slice_mean.content.flatten()[mask_zero]
        y_err = np.sqrt(theta_slice_rms.content.flatten()[mask_zero]**2 - y_fit**2)
        model = PowerLaw()
        model.prefactor.freeze(np.pi/6)
        model.fit(x_fit, y_fit, sigma=y_err, absolute_sigma=True)
        plt.figure("theta vs eta diff ratio")
        plt.errorbar(x_fit, y_fit, yerr=y_err, fmt=".k")
        plt.xlabel("(eta1 - eta2) / (eta1 + eta2)")
        plt.ylabel("theta")
        model.plot(fit_output=True)
        plt.xscale("linear")
        plt.yscale("linear")
        plt.legend()

    plt.show()


def test_calculate_n():
    x = np.array([[1, 1, -1]])
    y = np.array([[1, 2, -1]])
    n0, n1 = calculate_n(x, y)
    assert np.array_equal(n0, np.array([[0, 1]]))
    assert np.array_equal(n1, np.array([[-1/np.sqrt(2), -1/np.sqrt(2)]]))

def test_versor():
    x = np.array([[0, 1, 0]])
    y = np.array([[0, 0, -1]])
    n = calculate_versor(x, y)
    expected = np.array([[1, -1]]) / np.sqrt(2)
    assert np.allclose(n, expected)

def test_calculate_eta():
    pha = np.array([60, 30, 10]).reshape(1, 3)
    eta1, eta2 = calculate_eta(pha)
    assert np.isclose(eta1, 0.3)
    assert np.isclose(eta2, 0.1)

test_calculate_n()
test_versor()
test_calculate_eta()

hxeta(npixels=2, numbins=20)