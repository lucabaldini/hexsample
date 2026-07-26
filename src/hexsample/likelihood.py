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
from typing import Tuple

import numpy as np
from numba import njit

# Precompute the constant for speed
LOG2PI = math.log(2 * math.pi)

@njit
def coordinates(x: float, y: float, xbin0: float, ybin0: float, bin_size: float,
                shape: Tuple[int, int]) -> Tuple[int, int, float, float]:
    """Map the (x, y) position to the corresponding bi-dimensional matrix indices,
    calculating both the integer part (ix0, iy0) and the fractional part (wx, wy).

    Arguments
    ---------
    x : float
        The x coordinate to map.
    y : float
        The y coordinate to map.
    xbin0 : float
        The x coordinate of the first bin center in the matrix.
    ybin0 : float
        The y coordinate of the first bin center in the matrix.
    bin_size : float
        The size of the bins in the matrix.
    shape : Tuple[int, int]
        The shape of the matrix.
    """
    nx, ny = shape
    # Find the integer pixel indices.
    x_frac = (x - xbin0) / bin_size
    y_frac = (y - ybin0) / bin_size
    ix0 = int(math.floor(x_frac))
    iy0 = int(math.floor(y_frac))
    # Calculate the fractional coordinates within the pixel.
    wx = x_frac - ix0
    wy = y_frac - iy0
    # Clamp the indices to be within the bounds of the map
    ix0 = max(0, min(ix0, nx - 2))
    iy0 = max(0, min(iy0, ny - 2))
    # Clamp the fractional coordinates to be within [0, 1]
    wx = max(0.0, min(wx, 1.0))
    wy = max(0.0, min(wy, 1.0))
    return ix0, iy0, wx, wy


@njit
def interpolation(f: np.ndarray, ix0: int, iy0: int, wx: float, wy: float) -> np.ndarray:
    """Perform bilinear interpolation of the value of the charge fraction for each
    of the seven pixels in the cluster, returning an array with the interpolated values.

    The interpolation is performed using the four nearest bins in the charge fraction
    matrix. To interpolate the value, the fractional coordinates (wx, wy) are used to
    weight the contributions of the four bins.

    Arguments
    ---------
    f : np.ndarray
        The charge fraction matrix, with shape (7, nx, ny)
    ix0 : int
        The integer x index of the lower-left bin in the interpolation.
    iy0 : int
        The integer y index of the lower-left bin in the interpolation.
    wx : float
        The fractional x coordinate within the pixel.
    wy : float
        The fractional y coordinate within the pixel.
    """
    # Allocate the array for the interpolated values.
    f_interp = np.zeros(7)
    # Loop over each pixel matrix.
    for i in range(7):
        # Get the value of the four bins used for the interpolation.
        v00 = f[i, ix0, iy0]
        v10 = f[i, ix0 + 1, iy0]
        v01 = f[i, ix0, iy0 + 1]
        v11 = f[i, ix0 + 1, iy0 + 1]
        # Calculate the interpolated value with bilinear interpolation,
        # using the fractional coordinates as weights.
        f_interp[i] =  (v00 * (1 - wx) * (1 - wy) +
                        v10 * wx * (1 - wy) +
                        v01 * (1 - wx) * wy +
                        v11 * wx * wy)
    return f_interp


@njit
def interpolation_derivatives(f: np.ndarray, ix0: int, iy0: int, wx: float, wy: float,
                              bin_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform bilinear interpolation of the value of the charge fraction for each
    of the seven pixels in the cluster, and compute the derivatives with respect to
    x and y.

    The interpolation is performed using the four nearest bins in the charge fraction
    matrix. To interpolate the value, the fractional coordinates (wx, wy) are used to
    weight the contributions of the four bins.

    Arguments
    ---------
    f : np.ndarray
        The charge fraction matrix, with shape (7, nx, ny)
    ix0 : int
        The integer x index of the lower-left bin in the interpolation.
    iy0 : int
        The integer y index of the lower-left bin in the interpolation.
    wx : float
        The fractional x coordinate within the pixel.
    wy : float
        The fractional y coordinate within the pixel.
    bin_size : float
        The size of each bin in the charge fraction matrix.
    """
    # Allocate the arrays for the interpolated values and their derivatives.
    f_interp = np.zeros(7)
    df_dx = np.zeros(7)
    df_dy = np.zeros(7)
    # Loop over each pixel matrix.
    for i in range(7):
        # Get the value of the four bins used for the interpolation.
        v00 = f[i, ix0, iy0]
        v10 = f[i, ix0 + 1, iy0]
        v01 = f[i, ix0, iy0 + 1]
        v11 = f[i, ix0 + 1, iy0 + 1]
        # Calculate the interpolated value with bilinear interpolation,
        # using the fractional coordinates as weights.
        f_interp[i] =  (v00 * (1 - wx) * (1 - wy) +
                        v10 * wx * (1 - wy) +
                        v01 * (1 - wx) * wy +
                        v11 * wx * wy)
        # Calculate the derivatives with respect to x and y using the differences
        # between the bins, weighted by the fractional coordinates.
        df_dx[i] = (v10 - v00) * (1 - wy) + (v11 - v01) * wy
        df_dy[i] = (v01 - v00) * (1 - wx) + (v11 - v10) * wx
    # Divide the derivatives by the bin size to get the correct units.
    df_dx /= bin_size
    df_dy /= bin_size
    return f_interp, df_dx, df_dy


@njit
def weighted_pha(pha: np.ndarray, f_interp: np.ndarray, inv_sigma2: np.ndarray) -> float:
    """Compute the summed pha that minimizes the negative log-likelihood for fixed (x, y).
    This allows to profile out the summed pha from the optimization, reducing the number of
    free parameters of the fit.
    
    Arguments
    ---------
    pha : np.ndarray
        The array of pulse heights for the 7 pixels in the cluster.
    f_interp : np.ndarray
        The array of interpolated charge fractions for the 7 pixels in the cluster.
    inv_sigma2 : np.ndarray
        The array of inverse noise variances for the 7 pixels in the cluster.
    """
    sum_qf = .0
    sum_f2 = .0
    # Calculate the weighted sums needed to compute the optimal energy.
    for i in range(7):
        sum_qf += pha[i] * f_interp[i] * inv_sigma2[i]
        sum_f2 += f_interp[i]**2 * inv_sigma2[i]
    # Use the max to avoid negative energy values.
    return max(.0, sum_qf / (sum_f2 + 1e-12))


@njit
def sherman_morrison_inverse(res: np.ndarray, f_interp: np.ndarray, total_pha: float,
                             noise: np.ndarray):
    """Compute the chi2 (res^T * V^-1 * res) and log-determinant of the covariance
    matrix V using the Sherman-Morrison formula and the matrix determinant lemma.

    This avoids the explicit inversion of the 7x7 matrix and allows to save space and time.
    """
    # First we compute the diagonal elements of D, given by the square sum of electronic
    # noise and statistical fluctuations, and their inverses.
    d_matrix = np.zeros(7)
    inv_d_matrix = np.zeros(7)
    for i in range(7):
        d_matrix[i] = (noise[i] ** 2) + total_pha * f_interp[i]
        inv_d_matrix[i] = 1.0 / d_matrix[i]
    # Then we calculate the term at the denominator.
    sum_fk2_d = 0.0
    for k in range(7):
        sum_fk2_d += (f_interp[k] ** 2) * inv_d_matrix[k]
    # The final denominator term.
    g = 1.0 / (1.0 - total_pha * sum_fk2_d + 1e-12)
    # Calculate the log determinant of V using the matrix determinant lemma.
    prod_d = 0.0
    for k in range(7):
        prod_d += math.log(d_matrix[k])
    log_det_v = prod_d + math.log(abs(1.0 - total_pha * sum_fk2_d) + 1e-12)
    # Calculate the quadratic form res^T * V^-1 * res using Sherman-Morrison algebra.
    diagional_term = 0.0
    cross_term = 0.0
    for i in range(7):
        diagional_term += (res[i] ** 2) * inv_d_matrix[i]
        cross_term += res[i] * f_interp[i] * inv_d_matrix[i]
    # Final chi2 value.
    chi2 = diagional_term + g * total_pha * (cross_term ** 2)
    # Calculate the vector w = V^-1 * res for the gradient calculation.
    alpha = g * total_pha * cross_term
    vettore_w = np.zeros(7)
    for i in range(7):
        vettore_w[i] = inv_d_matrix[i] * (res[i] + alpha * f_interp[i])
    dnll_df = np.zeros(7)
    for i in range(7):
        dnll_df[i] = total_pha * (
            0.5 * inv_d_matrix[i]
            - vettore_w[i]
            - 0.5 * (vettore_w[i] ** 2)
            + g * vettore_w[i] * f_interp[i] * (vettore_w[i] - inv_d_matrix[i] * res[i])
        )
    return chi2, log_det_v, dnll_df


@njit
def nll_numba(x: float, y: float, total_pha: float, pha: np.ndarray, f: np.ndarray,
              xbin0: float, ybin0: float, bin_size: float, noise: np.ndarray) -> float:
    """Compute the negative log-likelihood for a given position (x, y).

    The model is based on the Gaussian diffusion of the charge cloud, and uses the precomputed
    charge fractions in each pixel from the f map.
    """
    # Calculate the bin indices and fractional coordinates for the interpolation
    ix0, iy0, wx, wy = coordinates(x, y, xbin0, ybin0, bin_size, f.shape[1:])
    # Interpolate the charge fractions for the 7 pixels in the cluster
    f_interp = interpolation(f, ix0, iy0, wx, wy)
    # We need to invert the covariance matrix V. We use the Sherman-Morrison formula
    # to avoid inverting a full 7x7 matrix. This is possible because V is a diagonal
    # matrix D plus a rank 1 matrix.
    res = np.zeros(7)
    for i in range(7):
        res[i] = pha[i] - (total_pha * f_interp[i])
    chi2, log_det_v, _ = sherman_morrison_inverse(res, f_interp, total_pha, noise)
    # Finally we compute the negative log-likelihood using the chi2 and the log-determinant of V.
    nll = 0.5 * chi2 + 0.5 * log_det_v
    return nll


@njit
def nll_grad_numba(x: float, y: float, total_pha: float, pha: np.ndarray, f: np.ndarray,
                   xbin0: float, ybin0: float, bin_size: float, noise: np.ndarray) -> np.ndarray:
    """Compute the complete analytical gradient of the exact multivariate NLL with
    respect to x, y, and Q simultaneously (Returns array of size 3).
    """
    # Calculate the bin indices and fractional coordinates for the interpolation
    ix0, iy0, wx, wy = coordinates(x, y, xbin0, ybin0, bin_size, f.shape[1:])
    # Interpolate the charge fractions and their derivatives for the 7 pixels in the cluster
    f_interp, df_dx, df_dy = interpolation_derivatives(f, ix0, iy0, wx, wy, bin_size)
    # Calculate the residuals for the 7 pixels in the cluster.
    res = np.zeros(7)
    for i in range(7):
        res[i] = pha[i] - (total_pha * f_interp[i])
    # We need to invert the covariance matrix V. We use the Sherman-Morrison formula
    # to avoid inverting a full 7x7 matrix.
    _, _, dnll_df = sherman_morrison_inverse(res, f_interp, total_pha, noise)
    # Finally we compute the gradient of the negative log-likelihood with respect to x, y, and Q.
    gnll_x = 0.0
    gnll_y = 0.0
    for i in range(7):
        gnll_x += dnll_df[i] * df_dx[i]
        gnll_y += dnll_df[i] * df_dy[i]
    gnll_q = 0.0
    for i in range(7):
        gnll_q -= (res[i] * f_interp[i]) / (noise[i] ** 2)
    # Return the gradient as a numpy array of size 3.
    return np.array([gnll_x, gnll_y, gnll_q])
