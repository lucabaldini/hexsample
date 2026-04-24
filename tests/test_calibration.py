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

from hexsample.calibration import CalibrationMatrix


def test_initialization_matrix():
    """Test the initialization of the CalibrationMatrix class.
    """
    # Create the calibration matrix object
    shape = (10, 10)
    cal_matrix = CalibrationMatrix(*shape)
    # Check initial state of the matrix and hits
    assert cal_matrix.shape == shape
    assert np.array_equal(cal_matrix.matrix, np.full(shape, np.nan), equal_nan=True)
    assert np.array_equal(cal_matrix.hits, np.zeros(shape, dtype=int))
    # Now test the matrix setters
    new_matrix = np.full(shape, 1.)
    cal_matrix.matrix = new_matrix
    assert np.array_equal(cal_matrix.matrix, new_matrix)
    # Test hits modification
    cal_matrix.hits[0, 0] += 1
    assert cal_matrix.hits[0, 0] == 1
    # Test the __call__ method
    assert cal_matrix(0, 0) == cal_matrix.matrix[0, 0]
    assert np.array_equal(cal_matrix([0, 1], [0, 1]), np.array([1., 1.]))

def test_set_value_matrix():
    """Test the set_value method of the CalibrationMatrix class.
    """
    shape = (5, 5)
    value = 2.
    cal_matrix = CalibrationMatrix(*shape)
    cal_matrix.set_value(value)
    assert np.array_equal(cal_matrix.matrix, np.full(shape, 2.))

def test_fill_matrix():
    """Test the fill method of the CalibrationMatrix class.
    """
    shape = (5, 5)
    value = 1.
    cal_matrix = CalibrationMatrix(*shape)
    cal_matrix.set_value(value)
    cal_matrix.hits[0, :] = 10
    cal_matrix.fill(2., max_hits=5)
    assert np.array_equal(cal_matrix.matrix[1:, :], np.full((4, 5), 2.))
