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
    if x.shape[1] == 3:
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
    if npixels == 3:
        cos_theta = np.sum(photon_pos * n, axis=1) / r
        theta = np.arccos(cos_theta)
        plt.figure("theta")
        plt.hist(theta, bins=100)

        eta1, eta2 = calculate_eta(pha)
        x = eta1 + eta2
        x_r_binning = np.linspace(0., 2/3, numbins + 1)
        x_theta_binning = np.linspace(0.1, 2/3, 2)
        y = (eta1 - eta2) / (eta1 + eta2)
        y_binning = np.linspace(0., 1, numbins + 1)

        xlabel = "eta1 + eta2"
        ylabel = "(eta1 - eta2) / (eta1 + eta2)"
        r_binning = np.linspace(0., 1/np.sqrt(3), 101)
        r_hist3d = Histogram3d(x_r_binning, y_binning, r_binning, xlabel=xlabel, ylabel=ylabel, zlabel="r")
        r_hist3d.fill(x, y, r)
        r_mean, r_rms = r_hist3d.collapse_axis(2)
        plt.figure("r mean")
        r_mean.plot()

        theta_binning = np.linspace(0, np.pi/6, 101)
        theta_hist3d = Histogram3d(x_r_binning, y_binning, theta_binning, xlabel=xlabel, ylabel=ylabel, zlabel="theta")
        theta_hist3d.fill(x, y, theta)
        theta_mean, theta_rms = theta_hist3d.collapse_axis(2)
        plt.figure("theta mean")
        theta_mean.plot()

        theta_slice = Histogram3d(x_theta_binning, y_binning, theta_binning, xlabel=xlabel, ylabel=ylabel, zlabel="theta")
        theta_slice.fill(x, y, theta)
        theta_slice_mean, theta_slice_rms = theta_slice.collapse_axis(2)
        plt.figure("theta slice mean")
        theta_slice_mean.plot()
        plt.figure("theta slice rms")
        theta_slice_rms.plot()
        x_fit = theta_slice_mean.bin_centers(axis=1)[:-1]
        y_fit = theta_slice_mean.content[0][:-1]
        y_err = np.sqrt(theta_slice_rms.content[0][:-1]**2 - y_fit**2)
        plt.figure("theta vs eta diff ratio")
        plt.errorbar(x_fit, y_fit, yerr=y_err, fmt=".k",
                     label='theta vs (eta1 - eta2)/(eta1 + eta2)')
        plt.xlabel("(eta1 - eta2) / (eta1 + eta2)")
        plt.ylabel("theta")
        
        def fit_model(x, a):
            return np.pi/6 * (1-  x**a)
        popt, pcov = curve_fit(fit_model, x_fit, y_fit, sigma=y_err, absolute_sigma=True)
        plt.plot(x_fit, fit_model(x_fit, popt[0]), 'r--', label=f'fit: a={popt[0]:.2f}')
        plt.legend()

    plt.show()




def hxeta_old(npixels = 3, numbins= 10):
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

    pha = np.array(pha)
    x = np.array(x)
    y = np.array(y)
    absx = np.array(absx)
    absy = np.array(absy)

    eta_vals = calculate_eta(pha)
    n = calculate_n(x, y)
    photon_pos = np.array([absx - x[:, 0], absy - y[:, 0]]).T / header['pitch']
    if npixels == 2:
        du = abs(np.sum(photon_pos * n, axis=1))
    elif npixels == 3:
        nu, nv = n
        n_center = (nu + nv) / np.linalg.norm(nu+nv, axis=1, keepdims=True)
        # n_center = nu
        du = abs(np.sum(photon_pos * nu, axis=1))
        dv = abs(np.sum(photon_pos * nv, axis=1))
        r = np.sqrt(photon_pos[:, 0]**2 + photon_pos[:, 1]**2)
        sign_mask = dv > du
        theta = np.arccos((photon_pos[:, 0]*n_center[:, 0] + photon_pos[:, 1]*n_center[:, 1]) / np.linalg.norm(photon_pos, axis=1))
        theta[sign_mask] = -theta[sign_mask]

    plt.figure()
    plt.hist(theta, bins=100)      
    # Plot eta function
    plt.figure(figsize=(8, 6))
    if npixels == 2:
        plt.scatter(eta_vals, du, alpha=0.5, s=1)
    if npixels == 3:
        eta1, eta2 = eta_vals
        # plt.scatter(eta1, dv, alpha=0.5, s=1, label='eta1 vs du')
        # plt.scatter(eta2, dv, alpha=0.5, s=1, label='eta2 vs du')
        # plt.xlabel("eta1, eta2")
        # plt.ylabel("du")
        # plt.legend()
        # plt.figure()
        # plt.hist(eta1, bins=20)
        # plt.xlabel("eta1")
        # plt.figure()
        # plt.hist(eta2, bins=20)
        # plt.xlabel("eta2")
        # plt.figure()
        # plt.scatter(eta1, eta2, alpha=0.5, s=1)
        # eta1_edges = np.linspace(min(eta1), max(eta1), numbins + 1)
        # eta2_edges = np.linspace(min(eta2), max(eta2), numbins + 1)
        # hist = Histogram2d(eta1_edges, eta2_edges)
        # hist.fill(eta1, eta2)
        # plt.figure()
        # hist.plot()

        x = eta1 + eta2
        x_r_binning = np.linspace(0., 2/3, numbins + 1)
        x_theta_binning = np.linspace(0.1, 2/3, 2)
        y = (eta1 - eta2) / (eta1 + eta2)
        y_binning = np.linspace(0., 1, numbins + 1)
        y_centers = 0.5 * (y_binning[:-1] + y_binning[1:])

        r_binning = np.linspace(0., 1/np.sqrt(3), 101)
        theta_binning = np.linspace(-np.pi/6, np.pi/6, 101)

        r_hist3d_mean, _ = hist_mean(x, y, r, x_r_binning, y_binning, r_binning)
        theta_hist3d_mean, _ = hist_mean(x, y, theta, x_r_binning, y_binning, theta_binning)

        plt.figure("r mean")
        plt.imshow(r_hist3d_mean.T, extent=(x_r_binning[0], x_r_binning[-1], y_binning[0], y_binning[-1]), origin='lower')
        plt.xlabel("eta1 + eta2")
        plt.ylabel("(eta1 - eta2) / (eta1 + eta2)")
        plt.colorbar()

        plt.figure("theta mean")
        plt.imshow(theta_hist3d_mean.T, extent=(x_r_binning[0], x_r_binning[-1], y_binning[0], y_binning[-1]), origin='lower')
        plt.xlabel("eta1 + eta2")
        plt.ylabel("(eta1 - eta2) / (eta1 + eta2)")
        plt.colorbar()

        theta_slice, std = hist_mean(x, y, theta, x_theta_binning, y_binning, theta_binning)
        print(std)
        plt.figure("theta vs eta diff ratio")
        plt.errorbar(y_centers, theta_slice[0], yerr=std[0], fmt=".k",
                     label='theta vs (eta1 - eta2)/(eta1 + eta2)')
        plt.xlabel("(eta1 - eta2) / (eta1 + eta2)")
        plt.ylabel("theta")
        yy = np.linspace(0, 1, 100)
        # plt.plot(yy, np.pi/6*(1 - 6/np.pi*np.arctan(1/np.sqrt(3)*yy))**0.6, 'k--')
        # plt.plot(yy, np.pi/6*(1-yy)**(0.5), 'r--')
        plt.plot(yy, np.pi/6*(1-yy)**(0.6), 'b--')
        plt.plot(yy, np.pi/6*yy**(1/0.6), 'g--')
        #mask the input data for fitting
        mask = ~np.isnan(theta_slice[0])
        def power_law_fit(x, a):
            return np.pi/6 * x**(a)
        popt, pcov = curve_fit(power_law_fit, y_centers[mask], theta_slice[0][mask], sigma=std[0][mask], absolute_sigma=True)
        plt.plot(yy, power_law_fit(yy, popt[0]),'r--', label=f'fit: a={popt[0]:.2f}')
        plt.legend()
        

        # plt.figure()
        # # for i in range(numbins):
        #     # plt.plot(eta_diff_ratio_binning[:-1], hist3d_mean, label=f'eta2 bin {i}')
        # plt.plot(eta_binning[:-1], hist3d_mean[0, :], label=f'eta diff last bin')
        # plt.plot(eta_binning[:-1], PowerLaw.evaluate(eta_binning[:-1]/0.5, 0.53, 0.272), '--')
        # plt.plot(eta_binning[:-1], PowerLaw.evaluate(eta_binning[:-1]/0.5, 0.5, 0.272), 'k--')

        plt.legend()

    plt.show()
# hxeta(3, 20)

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

hxeta(npixels=3, numbins=20)