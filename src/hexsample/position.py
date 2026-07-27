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

"""Position reconstruction facilities.
"""

from typing import Tuple

import numpy as np
from aptapy.hist import Histogram2d
from aptapy.models import Probit
from iminuit import Minuit
from scipy.interpolate import PchipInterpolator

from .likelihood import nll_grad_numba, nll_numba


def profile(xdata: np.ndarray, ydata: np.ndarray, xbins: int, ybins: int
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    # Create the 2D histogram to compute the profile
    xedges = np.linspace(xdata.min(), xdata.max(), xbins + 1).flatten()
    yedges = np.linspace(ydata.min(), ydata.max(), ybins + 1).flatten()
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


def versor_2pix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate the unit vector of a two-pixel cluster. 
    
    This versor is defined as the versor pointing from the center of the
    higher-pha pixel to the center the lower-pha pixel.

    Arguments
    ---------
    x : np.ndarray
        The x coordinates of the centers of the two pixels.
    
    y : np.ndarray
        The y coordinates of the centers of the two pixels.
    
    Returns
    -------
    versor : np.ndarray
        The unit vector of the two-pixel cluster.
    """
    # Calculate the vector.
    versor = np.array([x[1] - x[0], y[1] - y[0]])
    # Normalize it to get the versor.
    norm = np.hypot(versor[0], versor[1])
    if norm == 0:
        return np.zeros(2)
    return versor / norm


def versor_3pix(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the unit vectors of a three-pixel cluster.
    
    The first versor points from the center of the higher-pha pixel to the midpoint
    between the two lower-pha pixels. The second is orthogonal to the first one, and
    its direction is chosen to point towards the second higher-pha pixel.

    Arguments
    ---------
    x : np.ndarray
        The x coordinates of the centers of the three pixels.

    y : np.ndarray
        The y coordinates of the centers of the three pixels.

    Returns
    -------
    u : np.ndarray
        The first unit vector of the three-pixel cluster.
    
    v : np.ndarray
        The second unit vector of the three-pixel cluster.
    """
    # Calculate the first vector.
    u = np.array([x[1] + x[2] - 2 * x[0], y[1] + y[2] - 2 * y[0]])
    # Normalize it to get the versor.
    norm = np.hypot(u[0], u[1])
    if norm == 0:
        return np.zeros(2), np.zeros(2)
    u /= norm
    # Calculate the second vector as the orthogonal to the first one.
    v = np.array([-u[1], u[0]])
    # Verify that it point towards the second higher-pha pixel. If not, invert it.
    if (x[1] - x[0]) * v[0] + (y[1] - y[0]) * v[1] < 0:
        v = -v
    return u, v


def eta_2pix(
    pha: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    two_pix_rad_sigma: float
) -> Tuple[float, float]:
    """Calculate the incident position of the photon in a two-pixel cluster
    using the eta reconstruction algorithm.

    Arguments
    ---------
    pha : np.ndarray
        The measured pha in the two pixels of the cluster, ordered in
        decreasing order of pha.
    
    x : np.ndarray
        The x coordinates of the centers of the two pixels.

    y : np.ndarray
        The y coordinates of the centers of the two pixels.

    two_pix_rad_sigma : float
        The sigma parameter of the Probit function used to calculate the distance
        from the higher-pha pixel center. This parameter is expressed in units of
        the pixel pitch.
    
    Returns
    -------
    dx : float
        The x coordinate of the incident position of the photon, relative to the
        center of the higher-pha pixel and expressed in units of the pixel pitch.

    dy : float
        The y coordinate of the incident position of the photon, relative to the
        center of the higher-pha pixel and expressed in units of the pixel pitch.
    """
    # Calculate the eta variable, defined as the ratio between the lower-pha pixel
    # and the total pha in the cluster.
    eta = pha[1] / np.sum(pha)
    # Calculate the distance from the center of the higher-pha pixel, using the
    # Probit function to model the charge diffusion around the incident position.
    dr = Probit.evaluate(eta, 0.5, two_pix_rad_sigma)
    # Calculate the versor of the two-pixel cluster.
    versor = versor_2pix(x, y)
    # Project the radial coordinate onto the x and y axes to get the coordinates
    # of the incident position relative to the center of the higher-pha pixel.
    dx, dy = dr * versor[0], dr * versor[1]
    return dx, dy


def eta_2pix_unmodeled(
    pha: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    spline_2pix: PchipInterpolator
) -> Tuple[float, float]:
    """Calculate the incident position of the photon in a two-pixel cluster
    using the eta reconstruction algorithm with unmodeled charge diffusion.

    Arguments
    ---------
    pha : np.ndarray
        The measured pha in the two pixels of the cluster, ordered in
        decreasing order of pha.
    
    x : np.ndarray
        The x coordinates of the centers of the two pixels.

    y : np.ndarray
        The y coordinates of the centers of the two pixels.

    spline_2pix : PchipInterpolator
        The spline used to model the charge diffusion around the incident position.

    Returns
    -------
    dx : float
        The x coordinate of the incident position of the photon, relative to the
        center of the higher-pha pixel and expressed in units of the pixel pitch.

    dy : float
        The y coordinate of the incident position of the photon, relative to the
        center of the higher-pha pixel and expressed in units of the pixel pitch.
    """
    # Calculate the eta variable, defined as the ratio between the lower-pha pixel
    # and the total pha in the cluster.
    eta = pha[1] / np.sum(pha)
    # Calculate the distance from the center of the higher-pha pixel, using the
    # spline to model the charge diffusion around the incident position.
    dr = spline_2pix(eta)
    # Calculate the versor of the two-pixel cluster.
    versor = versor_2pix(x, y)
    # Project the radial coordinate onto the x and y axes to get the coordinates
    # of the incident position relative to the center of the higher-pha pixel.
    dx, dy = dr * versor[0], dr * versor[1]
    return dx, dy


def eta_3pix(
    pha: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    eta_3pix_rad_offset: float,
    eta_3pix_rad_sigma: float,
    eta_3pix_theta_sigma: float
) -> Tuple[float, float]:
    """Calculate the incident position of the photon in a three-pixel cluster
    using the eta reconstruction algorithm.

    Arguments
    ---------
    pha : np.ndarray
        The measured pha in the three pixels of the cluster, ordered in
        decreasing order of pha.
    
    x : np.ndarray
        The x coordinates of the centers of the three pixels.
    
    y : np.ndarray
        The y coordinates of the centers of the three pixels.
    
    eta_3pix_rad_offset : float
        The offset parameter of the Probit function used to calculate the distance
        from the higher-pha pixel center. This parameter is expressed in units of
        the pixel pitch.

    eta_3pix_rad_sigma : float
        The sigma parameter of the Probit function used to calculate the distance
        from the higher-pha pixel center. This parameter is expressed in units of
        the pixel pitch.
    
    eta_3pix_theta_sigma : float
        The sigma parameter of the Probit function used to calculate the angle of
        the incident position. This parameter is expressed in units of the pixel
        pitch.
    
    Returns
    -------
    dx : float
        The x coordinate of the incident position of the photon, relative to the
        center of the higher-pha pixel and expressed in units of the pixel pitch.

    dy : float
        The y coordinate of the incident position of the photon, relative to the
        center of the higher-pha pixel and expressed in units of the pixel pitch.
    """
    # Calculate the eta variables for the two lower-pha pixels, defined as the
    # ratio between the pha in each pixel and the total pha in the cluster.
    eta_1, eta_2 = pha[1:] / np.sum(pha)
    # Calculate the two new eta variables, used to model the radial and angular
    # coordinates of the incident position.
    eta_sum = eta_1 + eta_2
    eta_diff = (eta_1 - eta_2) / eta_sum
    # Calculate the radial and angular coordinates of the incident position using
    # the Probit function.
    r = Probit.evaluate(3 / 4 * eta_sum, eta_3pix_rad_offset, eta_3pix_rad_sigma)
    theta = Probit.evaluate((eta_diff + 1) / 2, 0., eta_3pix_theta_sigma) / r
    # Calculate the unit vectors for the radial and angular coordinates. The first
    # versor points from the center of the higher-pha pixel to the midpoint between
    # the two lower-pha pixels, while the second is orthogonal to the first one,
    # and its direction is chosen to point towards the second higher-pha pixel.
    u, v = versor_3pix(x, y)
    # Project the radial coordinate onto the x and y axes using the two versors
    # and the angle.
    dx = r * (np.cos(theta) * u[0] + np.sin(theta) * v[0])
    dy = r * (np.cos(theta) * u[1] + np.sin(theta) * v[1])
    return dx, dy


def eta_3pix_unmodeled(
    pha: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    spline_3pix_rad: PchipInterpolator,
    spline_3pix_theta: PchipInterpolator
) -> Tuple[float, float]:
    """Calculate the incident position of the photon in a three-pixel cluster
    using the eta reconstruction algorithm with unmodeled charge diffusion.

    Arguments
    ---------
    pha : np.ndarray
        The measured pha in the three pixels of the cluster, ordered in
        decreasing order of pha.
    
    x : np.ndarray
        The x coordinates of the centers of the three pixels.
    
    y : np.ndarray
        The y coordinates of the centers of the three pixels.
    
    spline_3pix_rad : PchipInterpolator
        The spline used to model the radial coordinate of the incident position.

    spline_3pix_theta : PchipInterpolator
        The spline used to model the angular coordinate of the incident position.

    Returns
    -------
    dx : float
        The x coordinate of the incident position of the photon, relative to the
        center of the higher-pha pixel and expressed in units of the pixel pitch.

    dy : float
        The y coordinate of the incident position of the photon, relative to the
        center of the higher-pha pixel and expressed in units of the pixel pitch.
    """
    # Calculate the eta variables for the two lower-pha pixels, defined as the
    # ratio between the pha in each pixel and the total pha in the cluster.
    eta_1, eta_2 = pha[1:] / np.sum(pha)
    # Calculate the two new eta variables, used to model the radial and angular
    # coordinates of the incident position.
    eta_sum = eta_1 + eta_2
    eta_diff = (eta_1 - eta_2) / eta_sum
    # Calculate the unit vectors for the radial and angular coordinates. The first
    # versor points from the center of the higher-pha pixel to the midpoint between
    # the two lower-pha pixels, while the second is orthogonal to the first one, and
    # its direction is chosen to point towards the second higher-pha pixel.
    u, v = versor_3pix(x, y)
    # Calculate the radial and angular coordinates of the incident position using
    # the splines to model the charge diffusion around the incident position.
    dr = spline_3pix_rad(eta_sum)
    dtheta = spline_3pix_theta(eta_diff)
    # Project the radial coordinate onto the x and y axes using the two versors
    # and the angle.    
    dx = dr * u[0] + dtheta * v[0]
    dy = dr * u[1] + dtheta * v[1]
    return dx, dy


def mle(
    pha: np.ndarray,
    noise: np.ndarray,
    f: np.ndarray,
    bin_size: float,
    xlims: Tuple[float, float],
    ylims: Tuple[float, float],
    p0: Tuple[float, float] = (0.0, 0.0),
) -> Minuit:
    """Perform maximum likelihood estimation of the incident position of the
    photon, given the observed pha in the 7 pixels of the cluster.

    The likelihood used for the fit is based on the Gaussian diffusion of the
    charge cloud around the incident position, and uses the precomputed charge
    fractions in each pixel to evaluate the likelihood for a given position.

    To speed up the computation, the negative log-likelihood and its gradient
    are implemented in the likelihood.py module and decorated with numba.njit.

    Arguments
    ---------
    pha : np.ndarray
        The measured pha in the 7 pixels of the cluster, ordered according to
        the convention defined in calibration.py.

    noise : np.ndarray
        The array of shape (7,) containing the equalized noise standard deviation
        for each pixel.
    
    f : np.ndarray
        The array of shape (7, nx, ny) containing the precomputed charge fractions
        in each pixel as a function of the incident position.

    bin_size : float
        The size of the bins in the f array, expressed in units of the pixel
        pitch.

    xlims : Tuple[float, float]
        The limits for the x coordinate of the f array, expressed in
        units of the pixel pitch.
    
    ylims : Tuple[float, float]
        The limits for the y coordinate of the f array, expressed in
        units of the pixel pitch.
    
    p0 : Tuple[float, float], optional
        The initial guess for the (x, y) position of the photon, expressed in
        units of the pixel pitch. A reasonable initial guess can be the centroid
        of the cluster. Default is the center of the pixel (0.0, 0.0).
    
    Returns
    -------
    m : Minuit
        The minimizer object containing all the information about the fit.
    """
    # Unpack the grid limits on the x and y axes.
    xmin, xmax = xlims
    ymin, ymax = ylims
    # Define the objective functions for the optimization, which are the
    # negative log-likelihood...
    def nll(x: float, y: float, q: float) -> float:
        return nll_numba(x, y, q, pha, f, xmin, ymin, bin_size, noise)
    # ... and its gradient.
    def nll_grad(x: float, y: float, q: float) -> Tuple[float, float]:
        return nll_grad_numba(x, y, q, pha, f, xmin, ymin, bin_size, noise)
    # Assign a name to the free parameters.
    parnames = ["x", "y", "q"]
    # Initialize the minimizer.
    p0 = (*p0, np.sum(pha))
    m = Minuit(nll, *p0, grad=nll_grad, name=parnames)
    # Set the limits for the free parameters.
    m.limits["x"] = (xmin, xmax)
    m.limits["y"] = (ymin, ymax)
    m.limits["q"] = (np.sum(pha) * 0.5, np.sum(pha) * 1.5)
    # Set the initial step sizes for the minimizer. We choose half the bin
    # size as default value. This value is automatically adjusted by the
    # minimizer during the fit, so it is not critical to choose a very
    # precise value.
    m.errors["x"] = bin_size / 2
    m.errors["y"] = bin_size / 2
    m.errors["q"] = np.sum(pha) * 0.1
    # Define the strategy for the minimizer. We use the higher strategy to
    # avoid numerical problems with the hessian. For an explanation of the
    # strategy levels, please refer to the iminuit documentation.
    m.strategy = 2
    # Run the minimization.
    m.migrad()
    # Return the minimizer object, which contains the best-fit values and
    # their uncertainties. It's better to return the whole object to allow
    # the caller to inspect the fit results and diagnostics.
    return m
