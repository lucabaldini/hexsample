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

"""Test suite for hexsample.calibration
"""

import numpy as np

from hexsample.calibration import CalibrationMatrixGain, CalibrationMatrixNoise

def test_matrices():
    """Test the setter of the calibration matrix.
    """
    num_cols, num_rows = 10, 10
    energy = 1000.
    # Test the default value of the matrix.
    gain = CalibrationMatrixGain(num_cols, num_rows, energy, default=None)
    assert gain.matrix.shape == (num_rows, num_cols)
    assert np.all(gain.matrix == 0.)
    gain = CalibrationMatrixGain(num_cols, num_rows, energy, default=0.5)
    assert gain.default == 0.5
    assert np.all(gain.matrix == 0.5)
    # Test the setter of the matrix.
    new_matrix = np.full((num_rows, num_cols), 1.)
    gain.matrix = new_matrix
    assert np.all(gain.matrix == 1.)
    # The default value does not change when the matrix is updated.
    assert gain.default == 0.5
    # Test the noise matrix.
    noise = CalibrationMatrixNoise(num_cols, num_rows, default=None)
    assert noise.matrix.shape == (num_rows, num_cols)
    assert np.all(noise.matrix == 0.)
    noise = CalibrationMatrixNoise(num_cols, num_rows, default=1.)
    assert noise.default == 1.
    assert np.all(noise.matrix == 1.)
    new_matrix = np.full((num_rows, num_cols), 0.5)
    noise.matrix = new_matrix
    assert np.all(noise.matrix == 0.5)
    assert noise.default == 1.


