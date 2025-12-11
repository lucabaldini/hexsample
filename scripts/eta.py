"""Analyze eta function and fit with power law
"""

from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

from aptapy.hist import Histogram2d

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


def profile_2d(d, eta1, eta2, xedges, yedges):
    vals = np.zeros((len(xedges)-1, len(yedges)-1))
    std = np.zeros((len(xedges)-1, len(yedges)-1))
    x_consecutive = list(zip(xedges[:-1], xedges[1:]))
    y_consecutive = list(zip(yedges[:-1], yedges[1:]))

    for i, (xlow, xhigh) in enumerate(x_consecutive):
        for j, (ylow, yhigh) in enumerate(y_consecutive):
            mask = (eta1 >= xlow) & (eta1 < xhigh) & (eta2 >= ylow) & (eta2 < yhigh)
            bin_data = d[mask]
            # print(len(bin_data))
            vals[i, j] =  np.mean(bin_data) if len(bin_data) > 0 else None 
            std[i, j] = np.std(bin_data) if len(bin_data) > 0 else None
    return vals, std

def power_law_2d(eta, gamma1, gamma2, gamma3):
    N = eta.size // 2
    eta1 = eta[:N]
    eta2 = eta[N:]
    u = 0.5 * ((eta1 / 0.5)**gamma1 + (eta2 / 0.5)**gamma2)
    v = 0.5 * ((eta2 / 0.5)**gamma3)
    return np.concatenate((u, v))

def hxeta(npixels = 3, numbins= 10):
    input_file_path = HEXSAMPLE_DATA / "hxsim_large.h5"
    input_file = DigiInputFileCircular(input_file_path)
    header = input_file.header
    args = HexagonalLayout(header['layout']), header['numcolumns'], header['numrows'],\
        header['pitch'], header['noise'], header['gain']
    readout = HexagonalReadoutCircular(*args)
    zsupthreshold = 10
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
        du = abs(np.sum(photon_pos * nu, axis=1))
        dv = abs(np.sum(photon_pos * nv, axis=1))
    
    # Plot eta function
    plt.figure(figsize=(8, 6))
    if npixels == 2:
        plt.scatter(eta_vals, du, alpha=0.5, s=1)
    if npixels == 3:
        eta1, eta2 = eta_vals
        plt.scatter(eta1, du, alpha=0.5, s=1, label='eta1 vs du')
        plt.scatter(eta2, dv, alpha=0.5, s=1, label='eta2 vs dv')
        plt.legend()

    if npixels == 3:
        eta1_edges = np.linspace(min(eta1), max(eta1), numbins + 1)
        eta2_edges = np.linspace(min(eta2), max(eta2), numbins + 1)
        hist = Histogram2d(eta1_edges, eta2_edges)
        hist.fill(eta1, eta2)
        plt.figure()
        hist.plot()
        du_vals, du_std = profile_2d(du, eta1, eta2, eta1_edges, eta2_edges)
        dv_vals, dv_std = profile_2d(dv, eta1, eta2, eta1_edges, eta2_edges)

        print(du_vals.shape)

        plt.figure("du ")
        plt.imshow(du_vals, extent=(eta1_edges[0], eta1_edges[-1], eta2_edges[0], eta2_edges[-1]), origin='lower')
        plt.colorbar()
        plt.xlabel("eta1")
        plt.ylabel("eta2")
        plt.figure("dv vals")
        plt.imshow(dv_vals, extent=(eta1_edges[0], eta1_edges[-1], eta2_edges[0], eta2_edges[-1]), origin='lower')
        plt.xlabel("eta1")
        plt.ylabel("eta2")
        plt.colorbar()
        ETA1_GRID, ETA2_GRID = np.meshgrid(hist.bin_centers(axis=0), hist.bin_centers(axis=1), indexing='ij')
        X = np.concatenate((ETA1_GRID.flatten(), ETA2_GRID.flatten()))
        du_Y = np.concatenate((du_vals.flatten(), dv_vals.flatten()))
        sigma = np.concatenate((du_std.flatten(), dv_std.flatten()))
        mask = sigma > 0
        X = X[mask]
        du_Y = du_Y[mask]
        sigma = sigma[mask]

        popt, pcov = curve_fit(power_law_2d, X, du_Y, sigma=sigma)
        opt_y = power_law_2d(X, *popt)
        chisq = np.sum(((du_Y - opt_y) / sigma)**2)
        ndof = len(du_Y) - len(popt)
        print(chisq, ndof, chisq/ndof)
        print(popt)

    plt.show()
hxeta(3, 20)