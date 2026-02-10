# Copyright (C) 2023--2025 the hexsample team.
#
# For the license terms see the file LICENSE, distributed along with this
# software.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""Calibration facilities.
"""

import numpy as np
from aptapy.hist import Histogram2d
from aptapy.models import Probit
from aptapy.plotting import last_line_color, plt
from tqdm import tqdm

from hexsample.clustering import ClusteringNN

from .fileio import DigiInputFileBase


def profile(xdata: np.ndarray, ydata: np.ndarray, xbins: int, ybins: int
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the profile of a set of xdata and ydata. The profile is computed by creating
    the 2D histogram and computing the median of the y-axis distribution for each x-bin.

    Arguments
    ---------
    xdata : np.ndarray
        The x data to be profiled.
    ydata : np.ndarray
        The y data to be profiled.
    xbins : int
        The number of bins in the x axis.
    ybins : int
        The number of bins in the y axis.

    Returns
    -------
        x : np.ndarray
            The bin centers in the x axis.
        y : np.ndarray
            The median values in the y axis for each x bin.
        yerr: np.ndarray
            The errors of the median values in the y axis for each x bin.
    """
    # Be sure that the input data are float arrays
    xdata = xdata.astype(float)
    ydata = ydata.astype(float)
    # Create the 2D histogram to compute the profile
    xedges = np.linspace(xdata.min(), xdata.max(), xbins + 1)
    yedges = np.linspace(ydata.min(), ydata.max(), ybins + 1)
    hist = Histogram2d(xedges, yedges)
    hist.fill(xdata, ydata)
    # Create the arrays to store the profile values and their errors
    x = hist.bin_centers()
    y = np.zeros(xbins)
    yerr = np.zeros(xbins)
    for i in range(xbins):
        # Slice the histogram on the x-axis to get the y-axis distribution
        hist_slice = hist.slice1d(i)
        entries = hist_slice.content.sum()
        if entries == 0:
            # If the slice histogram is empty, set the profile value and error to NaN
            y[i] = np.nan
            yerr[i] = np.nan
            continue
        # Compute the profile value as the median of the y-axis distribution
        y[i] = hist_slice.ppf(0.5)
        # Compute the error of the sample median
        yerr[i] = 1.253 * hist_slice.binned_statistics()[1] / np.sqrt(entries)
    # Remove the bins with NaN values and return the profile
    valid = ~np.isnan(y)
    return x[valid], y[valid], yerr[valid]


def angle(pos: np.ndarray, versors: np.ndarray) -> np.ndarray:
    """Calculate the angle between the photon position and the versors of the cluster.

    Arguments
    ---------
    pos : np.ndarray
        The position of the photon with respect to the most charged pixel, in units of pitch.
    versors : np.ndarray
        The versors of the cluster.

    Returns
    -------
    angle : np.ndarray
        The angle between the photon position and the versors of the cluster, in radians.
    """
    # Calculate the projections of the photon position on the versors
    x_proj = np.sum(pos * versors[:, 0], axis=1)
    y_proj = np.sum(pos * versors[:, 1], axis=1)
    # Estimate the angle
    return np.arctan2(y_proj, x_proj)


def distance(pos: np.ndarray, projection_axis: np.ndarray | None = None) -> np.ndarray:
    """Calculate the distance of the photon from the center of the most charged pixel. If
    specified, project the distance on the given projection axis, given as a unit vector.

    Arguments
    ---------
    pos : np.ndarray
        The position of the photon with respect to the most charged pixel, in units of pitch.
    projection_axis : np.ndarray | None
        The axis on which to project the distance, given as a unit vector. If None,
        the distance is not projected. Default is None.
    
    Returns
    -------
    distance : np.ndarray
        The distance of the photon from the center of the most charged pixel. If a projection axis
        is specified, the distance is projected onto that axis.
    """
    if projection_axis is None:
        # This is useful for 3-pixel events
        return np.sqrt(np.sum(pos**2, axis=1))
    # This is useful for 2-pixel events
    return np.sum(pos * projection_axis, axis=1)


def calibration_data(input_file: DigiInputFileBase, clustering: ClusteringNN, pitch: float
                     ) -> tuple[np.ndarray, ...]:
    """Open the simulated input file and extract the data needed for the calibration of the eta
    function. The data are extracted only for 2-pixel and 3-pixel clusters. The resuling arrays
    need to be masked to select the desired cluster size before the calibration.

    Arguments
    ---------
    input_file : DigiInputFileBase
        The input file to be analyzed.
    clustering : ClusteringNN
        The clustering algorithm to be used to reconstruct the clusters.
    pitch : float
        The pixel pitch of the detector.
    
    Returns
    -------
    size : np.ndarray
        Array containing the size of the clusters for each event.
    photon_pos : np.ndarray
        Array containing the position of the photon with respect to the most charged pixel, in
        units of pitch, for each event.
    versors : np.ndarray
        Array containing the versors of the cluster for each event.
    eta : np.ndarray
        Array containing the eta values for each event.
    """
    # Create the arrays to store the data. These arrays are bigger than needed, but we don't know
    # how many 2 and 3 pixel events we have.
    n_max = len(list(input_file))
    size = np.zeros(n_max, dtype=int)
    photon_pos = np.zeros((n_max, 2))
    versors = np.zeros((n_max, 2, 2))
    eta = np.zeros(n_max, dtype=object)
    count = 0
    # Loop over the events and calculate the interesting quantities.
    for i, event in tqdm(enumerate(input_file)):
        cluster = clustering.run(event)
        # Analyze only 2-pixel and 3-pixel events.
        if cluster.size() == 2 or cluster.size() == 3:
            mc_event = input_file.mc_event(i)
            size[count] = cluster.size()
            # Calculate the photon position with respect to the most charged pixel
            ph_pos = np.array([mc_event.absx - cluster.x[0],
                               mc_event.absy - cluster.y[0]]) / pitch
            photon_pos[count] = ph_pos
            eta[count] = cluster.calculate_eta()
            versors[count] = cluster.versors()
            count += 1
    # Slice the arrays to remove the empty entries
    return size[:count], photon_pos[:count], versors[:count], eta[:count]


def calibrate_dr_2pix(eta: np.ndarray, dr: np.ndarray, nbins: int, **kwargs) -> float:
    """Calibrate the 2-pixel eta function using the distance projection on the line connecting
    the two highest pixels in the cluster. The eta function is fitted with a probit model with the
    offset fixed to 0.5.

    Arguments
    ---------
    eta : np.ndarray
        The eta values for the 2-pixel clusters.
    dr : np.ndarray
        The distance of the photon from the center of the most charged pixel, projected onto the
        line connecting the two pixels, in units of pitch.

    Returns
    -------
    sigma : float
        The best-fit value of the sigma parameter of the probit function.
    """
    # Calculate the profile of the data
    x, y, yerr = profile(eta, dr, nbins, 101)
    # Fit the data with a probit model with offset fixed to 0.5
    model = Probit()
    model.offset.freeze(0.5)
    model.fit(x, y, sigma=yerr, absolute_sigma=True)
    # Plot the results
    fig = plt.figure("calibration_eta_vs_dr_2pix")
    plt.errorbar(x, y, yerr=yerr, fmt=".k", label="Monte Carlo simulation")
    fit_label = "2-pixel events calibration\n" + fr"$\sigma$ = {model.sigma.ufloat()}"
    model.plot(label=fit_label, color=last_line_color())
    plt.xlabel(r"$\eta$")
    plt.ylabel("r/p")
    plt.legend()
    # Save the figure if requested
    if kwargs.get("save", False):
        fig_path = kwargs.get("path")
        fig.savefig(fig_path / "2pix_cal.pdf", format="pdf")
    # Test with a pivot
    eta_pivot = 0.0423
    xx = np.linspace(0, max(x), 100)
    yy = np.where(xx < eta_pivot, model(eta_pivot) / eta_pivot * xx, model(xx))
    plt.figure("test fit_2pix")
    plt.plot(xx, yy, label="Fitted model", color=last_line_color())
    plt.errorbar(x, y, yerr=yerr, fmt=".k", label="Monte Carlo simulation")
    return model.sigma.value


def calibrate_dr_3pix(eta: np.ndarray, dr: np.ndarray, nbins: int, **kwargs) -> tuple[float, float]:
    """Calibrate the radial component of the 3-pixel eta function, using the distance of the photon
    from the center of the most charged pixel. The eta function is fitted with a probit model.

    Arguments
    ---------
    eta : np.ndarray
        The eta values for the 3-pixel clusters.
    dr : np.ndarray
        The distance of the photon from the center of the most charged pixel, projected onto the
        line connecting the two pixels, in units of pitch.

    Returns
    -------
    offset : float
        The best-fit value of the offset parameter of the probit function.
    sigma : float
        The best-fit value of the sigma parameter of the probit function.
    """
    # Calculate the sum of the eta values for each event
    eta_sum = np.sum(eta, axis=1)
    # Calculate the profile of the data
    x, y, yerr = profile(eta_sum, dr, nbins, 101)
    # Fit with a probit model
    model = Probit()
    model.fit(x, y, sigma=yerr, absolute_sigma=True)
    # Plot the results
    fig = plt.figure("calibration_eta_vs_dr_3pix")
    plt.errorbar(x, y, yerr=yerr, fmt=".k", label="Monte Carlo simulation")
    fit_label = "3-pixel events radial calibration\n" + fr"$\sigma$ = {model.sigma.ufloat()}"
    fit_label += "\n" + fr"$\mu$ = {model.offset.ufloat()}"
    model.plot(label=fit_label, color=last_line_color())
    plt.xlabel(r"$\eta^+$")
    plt.ylabel("r/p")
    plt.legend()
    # Save the figure if requested
    if kwargs.get("save", False):
        fig_path = kwargs.get("path")
        fig.savefig(fig_path / "3pix_cal_radial.pdf", format="pdf")
    # Test with a pivot
    eta_pivot = 0.054
    xx = np.linspace(0, max(x), 100)
    yy = np.where(xx < eta_pivot, model(eta_pivot) / eta_pivot * xx, model(xx))
    plt.figure("test fit_3pix")
    plt.plot(xx, yy, label="Fitted model", color=last_line_color())
    plt.errorbar(x, y, yerr=yerr, fmt=".k", label="Monte Carlo simulation")
    return model.offset.value, model.sigma.value


def calibrate_theta_3pix(eta: np.ndarray, dr: np.ndarray, theta: np.ndarray, nbins: int,
                         **kwargs) -> float:
    """Calibrate the angular component of the 3-pixel eta function using the angle between the
    photon position and the versor pointing from the center of the most charged pixel to the
    midpoint between the two less charged pixels, and the radial distance from the center of the
    most charged pixel. The eta function is fitted with a probit model with the offset fixed to 0.

    Arguments
    ---------
    eta : np.ndarray
        The eta values for the 3-pixel clusters.
    dr : np.ndarray
        The distance of the photon from the center of the most charged pixel, projected onto the
        line connecting the two pixels, in units of pitch.
    theta : np.ndarray
        The angle between the photon position and the versors of the cluster, in radians.

    Returns
    -------
    sigma : float
        The best-fit value of the sigma parameter of the probit function.
    """
    # Calculate the transverse component
    delta_r = dr * theta
    # Calculate eta-
    eta_diff = (eta[:, 0] - eta[:, 1]) / np.sum(eta, axis=1)
    # Calculate the profile of the data
    x, y, yerr = profile(eta_diff, delta_r, nbins, 101)
    # Fit with a probit model with offset fixed to 0.
    model = Probit()
    model.offset.freeze(0.)
    # We have to traslate the data to match the model definition.
    model.fit((1 + x)/2, y, sigma=yerr, absolute_sigma=True)
    # Plot the results
    fig = plt.figure("calibration_eta_vs_theta_3pix")
    plt.errorbar(x, y, yerr=yerr, fmt=".k", label="Monte Carlo simulation")
    fit_label = "3-pixel events angular calibration\n"
    fit_label += fr"$\sigma$ = {model.sigma.ufloat()}"
    # Plot the model
    xx = np.linspace(0.5, max((1 + x)/2), 100)
    plt.plot(2*xx - 1, model(xx), label=fit_label, color=last_line_color())
    plt.xlabel(r"$\eta^-$")
    plt.ylabel(r"r$\theta$/p")
    plt.legend()
    # Save the figure if requested
    if kwargs.get("save", False):
        fig.savefig(kwargs.get("path") / "3pix_cal_angular.pdf", format="pdf")
    return model.sigma.value
