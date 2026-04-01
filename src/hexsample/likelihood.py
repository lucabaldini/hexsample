# Copyright (C) 2023--2026 the hexsample team.
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

"""Likelihood facility for MLE-based clustering.
"""

import math

import numpy as np
from numba import njit

# Some precomputed numerical constants
SQRT2PI = math.sqrt(2 * math.pi)
SQRT2 = math.sqrt(2)
LOG2PI = math.log(2 * math.pi)


@njit
def nll_numba(x: float, y: float, pha: np.ndarray, f_map: np.ndarray,
              x0: float, y0: float, dx_bin: float, dy_bin: float, sigma: float) -> float:
    """Compute the negative log-likelihood for a given position (x, y) and energy.
    
    The model is based on the Gaussian diffusion of the charge cloud, and uses the precomputed
    charge fractions in each pixel from the f_map.
    """
    # Find the pixel indices corresponding to the (x, y) position in the map grid
    ix = int((x - x0) / dx_bin)
    iy = int((y - y0) / dy_bin)
    # Clamp the indices to be within the bounds of the map
    nx, ny, _ = f_map.shape
    ix = max(0, min(ix, nx-1))
    iy = max(0, min(iy, ny-1))
    # Start computing the negative log-likelihood
    _nll = 0.0
    # Precompute some constants for efficiency
    inv_sigma2 = 1.0 / (sigma**2)
    inv_sigma_sqrt2 = 1.0 / (sigma * SQRT2)
    # Loop over the 7 pixels in the cluster and accumulate the negative log-likelihood
    # contributions
    for i in range(7):
        # Compute the expected mean of the signal in i-th pixel
        mu = f_map[ix, iy, i] * pha.sum()
        # For pixels signal above zero, use the Gaussian likelihood
        if pha[i] >= 0 or pha[i] < 0:
            res = pha[i] - mu
            _nll += 0.5 * (res**2 * inv_sigma2 + LOG2PI)
        # For pixels with zero signal, use the CDF of the Gaussian to compute the probability of
        # observing zero. 
        else:
            z = mu * inv_sigma_sqrt2
            prob = 0.5 * (1.0 - math.erf(z))
            _nll -= math.log(prob + 1e-20)
    return _nll


@njit
def nll_grad_numba(x: float, y: float, pha: np.ndarray, f_map: np.ndarray,
                   dx_map: np.ndarray, dy_map: np.ndarray, x0: float, y0: float, dx_bin: float,
                   dy_bin: float, sigma: float) -> np.ndarray:
    """Compute the gradient of the negative log-likelihood with respect to the free parameters.
    These compution is needed to speed up the optimization process when using iminuit.
    """
    # Find the pixel indices corresponding to the (x, y) position in the map grid
    ix = int((x - x0) / dx_bin)
    iy = int((y - y0) / dy_bin)
    # Clamp the indices to be within the bounds of the map
    nx, ny, _ = f_map.shape
    ix = max(0, min(ix, nx-1))
    iy = max(0, min(iy, ny-1))
    # Initialize the gradients for x, y, and energy to zero
    gnll_x, gnll_y = 0., 0.
    energy = pha.sum()
    # Precompute some constants for efficiency
    inv_sigma2 = 1.0 / (sigma**2)
    # Loop over the 7 pixels in the cluster and accumulate the gradient contributions
    for i in range(7):
        # Compute the expected mean of the signal in i-th pixel
        f_i = f_map[ix, iy, i]
        mu = f_i * energy
        # For pixels signal above zero, the gradient with respect to the mean is the derivative of
        # the Gaussian likelihood
        if pha[i] >= 0 or pha[i] < 0:
            d_loss_dmu = -(pha[i] - mu) * inv_sigma2
        # For pixels with zero signal, the gradient is computed using the derivative of the CDF of
        # the Gaussian distribution
        else:
            z = mu / sigma
            pdf_z = 1 / SQRT2PI * math.exp(-0.5 * z**2)
            cdf_neg_z = 0.5 * (1.0 - math.erf(mu / (sigma * SQRT2)))
            d_loss_dmu = pdf_z / (sigma * cdf_neg_z + 1e-10)
        # Use the chain rule to compute the contributions to the gradients of x, y, and energy
        gnll_x += d_loss_dmu * energy * dx_map[ix, iy, i]
        gnll_y += d_loss_dmu * energy * dy_map[ix, iy, i]
    return np.array([gnll_x, gnll_y])
