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
from iminuit import Minuit

from .likelihood import nll_grad_numba, nll_numba


def eta_2pix(
    pha: np.ndarray,
    eta_2pix_rad_sigma
) -> Tuple[float, float]:
    eta = pha[1] / (pha[0] + pha[1])
    





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
    def nll(x: float, y: float) -> float:
        return nll_numba(x, y, pha, f, xmin, ymin, bin_size, noise)
    # ... and its gradient.
    def nll_grad(x: float, y: float) -> Tuple[float, float]:
        return nll_grad_numba(x, y, pha, f, xmin, ymin, bin_size, noise)
    # Assign a name to the free parameters.
    parnames = ["x", "y"]
    # Initialize the minimizer.
    m = Minuit(nll, *p0, grad=nll_grad, name=parnames)
    # Set the limits for the free parameters.
    m.limits["x"] = (xmin, xmax)
    m.limits["y"] = (ymin, ymax)
    # Set the initial step sizes for the minimizer. We choose half the bin
    # size as default value. This value is automatically adjusted by the
    # minimizer during the fit, so it is not critical to choose a very
    # precise value.
    m.errors["x"] = bin_size / 2
    m.errors["y"] = bin_size / 2
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
