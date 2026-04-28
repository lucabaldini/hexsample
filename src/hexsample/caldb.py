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

"""Calibration database facilities.
"""

from enum import Enum
from typing import Tuple

import numpy as np

from .tasks import HEXSAMPLE_DATA
from .calibration import CalibrationMatrix


class CalibrationFeatures(Enum):

    """Enum class expressing the possible calibration features.
    """

    GAIN = "gain"
    NOISE = "noise"
    PEDESTAL = "pedestal"

    @classmethod
    def values(cls) -> Tuple[str, ...]:
        """Return a tuple with all the enum values.
        """
        return tuple(item.value for item in cls)


def create_response_file(feature: str, loc: float, distribution: str, scale: float, num_cols: int, num_rows: int):
    """Create a response file for a given feature and value.
    """
    # Create the instance to store the calibration matrix.
    cal_matrix = CalibrationMatrix(num_cols, num_rows)
    # Create the matrix with the given distribution and location.
    if distribution == "uniform":
        matrix = np.full((num_rows, num_cols), loc)
    elif distribution == "gaussian":
        matrix = np.random.normal(loc, scale, (num_rows, num_cols))
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    # Store the matrix in the calibration matrix instance.
    cal_matrix.matrix = matrix
    # Define the file name and save the calibration matrix to an HDF5 file.
    output_file_path = f"{HEXSAMPLE_DATA}/cal_{feature}.hdf5"    # Just a momentary placeholder
    cal_matrix.to_hdf5(output_file_path, feature, True)
