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

import numpy as np

from .calibration import CalibrationMatrix

class MapIntent:
    pass


def create_response_file(num_cols: int, num_rows: int, feature: str, value: float):
    """Create a response file for a given feature and value.
    """
    cal_matrix = CalibrationMatrix(num_cols, num_rows)
    matrix = np.full((num_rows, num_cols), value)
    cal_matrix.matrix = matrix
    output_file_path = MapIntent(feature)
    cal_matrix.to_hdf5(output_file_path, feature, True)
