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

# Precompute the constant for speed
LOG2PI = math.log(2 * math.pi)

@njit
def nll_numba(x: float, y: float, pha: np.ndarray, f: np.ndarray, xbin0: float, ybin0: float,
              dx_bin: float, dy_bin: float, sigma: float) -> float:
    """Compute the negative log-likelihood for a given position (x, y).

    The model is based on the Gaussian diffusion of the charge cloud, and uses the precomputed
    charge fractions in each pixel from the f map. The energy is profiled out to reduce the
    dimensionality of the optimization.
    """
    # Find the pixel indices corresponding to the (x, y) position in the map grid
    x_frac = (x - xbin0) / dx_bin
    y_frac = (y - ybin0) / dy_bin
    ix0 = int(math.floor(x_frac))
    iy0 = int(math.floor(y_frac))
    # Clamp the indices to be within the bounds of the map
    nx, ny, _ = f.shape
    ix0 = max(0, min(ix0, nx - 2))
    iy0 = max(0, min(iy0, ny - 2))
    # Compute the local coordinates within the bin for bilinear interpolation
    wx = x_frac - ix0
    wy = y_frac - iy0
    # Interpolate the f values for the 7 pixels in the cluster
    f_interp = np.zeros(7)
    sum_qf = .0
    sum_f2 = .0
    for i in range(7):
        v00 = f[i, ix0, iy0]
        v10 = f[i, ix0 + 1, iy0]
        v01 = f[i, ix0, iy0 + 1]
        v11 = f[i, ix0 + 1, iy0 + 1]
        # Bilinear interpolation
        fi = (v00 * (1 - wx) * (1 - wy) +
              v10 * wx * (1 - wy) +
              v01 * (1 - wx) * wy +
              v11 * wx * wy)
        f_interp[i] = fi
        # Compute the terms needed for profiling out the energy
        sum_qf += pha[i] * fi
        sum_f2 += fi * fi
    # Profile out the energy by finding the value that minimizes the NLL for fixed (x, y)
    energy_opt = max(.0, sum_qf / (sum_f2 + 1e-12))  # Avoid division by zero
    # Now compute the NLL using the optimal energy
    nll = 0.0
    inv_sigma2 = 1.0 / (sigma**2)
    for i in range(7):
        mu = f_interp[i] * energy_opt
        res = pha[i] - mu
        nll += 0.5 * (res**2 * inv_sigma2 + LOG2PI)
    return nll


@njit
def nll_grad_numba(x: float, y: float, pha: np.ndarray, f: np.ndarray, xbin0: float, ybin0: float,
                   dx_bin: float, dy_bin: float, sigma: float) -> np.ndarray:
    """Compute the gradient of the negative log-likelihood with respect to the free parameters.
    """
    # Find the pixel indices corresponding to the (x, y) position in the map grid
    x_frac = (x - xbin0) / dx_bin
    y_frac = (y - ybin0) / dy_bin
    ix0 = int(math.floor(x_frac))
    iy0 = int(math.floor(y_frac))
    # Clamp the indices to be within the bounds of the map
    nx, ny, _ = f.shape
    ix0 = max(0, min(ix0, nx - 2))
    iy0 = max(0, min(iy0, ny - 2))
    # Compute the local coordinates within the bin for bilinear interpolation
    wx = x_frac - ix0
    wy = y_frac - iy0
    # Interpolate the f values for the 7 pixels in the cluster and their derivatives
    f_interp = np.zeros(7)
    df_dx = np.zeros(7)
    df_dy = np.zeros(7)
    sum_qf = .0
    sum_f2 = .0
    for i in range(7):
        v00 = f[i, ix0, iy0]
        v10 = f[i, ix0 + 1, iy0]
        v01 = f[i, ix0, iy0 + 1]
        v11 = f[i, ix0 + 1, iy0 + 1]
        # Bilinear interpolation
        fi = (v00 * (1 - wx) * (1 - wy) +
              v10 * wx * (1 - wy) +
              v01 * (1 - wx) * wy +
              v11 * wx * wy)
        f_interp[i] = fi
        # Derivatives of the interpolated f with respect to x and y
        df_dx[i] = ((v10 - v00) * (1 - wy) + (v11 - v01) * wy) / dx_bin
        df_dy[i] = ((v01 - v00) * (1 - wx) + (v11 - v10) * wx) / dy_bin
        # Compute the terms needed for profiling out the energy
        sum_qf += pha[i] * fi
        sum_f2 += fi * fi
    # Profile out the energy by finding the value that minimizes the NLL for fixed (x, y)
    energy_opt = max(.0, sum_qf / (sum_f2 + 1e-12))  # Avoid division by zero
    # Now compute the gradient using the optimal energy
    gnll_x = 0.0
    gnll_y = 0.0
    inv_sigma2 = 1.0 / (sigma**2)
    for i in range(7):
        mu = f_interp[i] * energy_opt
        d_loss_dmu = -(pha[i] - mu) * inv_sigma2
        gnll_x += d_loss_dmu * energy_opt * df_dx[i]
        gnll_y += d_loss_dmu * energy_opt * df_dy[i]
    return np.array([gnll_x, gnll_y])
