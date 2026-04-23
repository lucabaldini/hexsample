# Copyright (C) 2023--2025 the hexsample team.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Test suite for hexsample.likelihood
"""

import numpy as np
from aptapy.plotting import plt

from hexsample.calibration import ChargeFractionMatrices
from hexsample.likelihood import nll_numba


def test_nll_numba(test_data_path):
    """Test the nll_numba function.
    """
    # Test a simple case, this pha is taken from an event with 20 ENC noise, offset set to 512.
    pha = np.array([1023, -2, 601, 17, -23, 39, 55])
    x = np.linspace(-0.5, 0.5, 100)
    y = np.linspace(-1/np.sqrt(3), 1/np.sqrt(3), 100)
    # Load the charge fraction matrices and extract the relevant attributes
    mle_table_path = test_data_path("test_mle_matrices.h5")
    charge_fraction_matrices = ChargeFractionMatrices.from_hdf5(mle_table_path)
    f = charge_fraction_matrices.matrices
    x_bins = charge_fraction_matrices.x_bins
    y_bins = charge_fraction_matrices.y_bins
    xbin0 = x_bins[0]
    ybin0 = y_bins[0]
    bin_size = x_bins[1] - x_bins[0]

    sigma = 20
    nll = np.zeros((len(x), len(y)))
    for i_x, _x in enumerate(x):
        for i_y, _y in enumerate(y):
            nll[i_x, i_y] = nll_numba(_x, _y, pha, f, xbin0, ybin0, bin_size, sigma)
    plt.figure("test_negative_log_likelihood")
    plt.imshow(nll.T, extent=(x[0], x[-1], y[0], y[-1]), origin="lower")
    plt.colorbar(label="Negative log-likelihood")
