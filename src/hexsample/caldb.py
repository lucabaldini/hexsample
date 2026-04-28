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

import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np

from .tasks import HEXSAMPLE_DATA
from .calibration import CalibrationMatrix


class CalibrationType(Enum, str):

    """Enum class expressing the possible calibration types.
    """

    ENC = "enc"
    PEDESTAL = "pedestal"
    NOISE = "noise"
    GAIN = "gain"

    @classmethod
    def values(cls) -> Tuple[str, ...]:
        """Return a tuple with all the enum values.
        """
        return tuple(item.value for item in cls)


class CalDB:

    """Simple calibration database implementation.
    """

    DEFAULT_DIR = pathlib.Path(__file__).parent.parent / "caldb"

    def __init__(self, root_dir: pathlib.Path = DEFAULT_DIR):
        self.root_dir = root_dir

    def _open(self, calibration_type: CalibrationType, designator: str) -> CalibrationMatrix:
        """Open the calibration file for the given designation and intent.
        """
        if pathlib.Path(designator).is_file():
            return CalibrationMatrix.from_hdf5(designator)
        file_path = self.root_dir / calibration_type / f"{designator}.h5"
        return CalibrationMatrix.from_hdf5(file_path)

    def open_enc(self, designator: str) -> CalibrationMatrix:
        """Open the ENC calibration file for the given designation.
        """
        return self._open(CalibrationType.ENC, designator)

    def open_pedestal(self, designator: str) -> CalibrationMatrix:
        """Open the pedestal calibration file for the given designation.
        """
        return self._open(CalibrationType.PEDESTAL, designator)

    def open_noise(self, designator: str) -> CalibrationMatrix:
        """Open the noise calibration file for the given designation.
        """
        return self._open(CalibrationType.NOISE, designator)

    def open_gain(self, designator: str) -> CalibrationMatrix:
        """Open the gain calibration file for the given designation.
        """
        return self._open(CalibrationType.GAIN, designator)



@dataclass
class MapCalibration:

    """Class to manage the calibration files.
    """

    intent: str
    loc: int
    chip: str = "xpol3"
    pattern: str = "uniform"
    distibution: str = "gauss"
    scale_ratio: float = 1.0
    sim: bool = False
    version: int = 1

    def filename(self) -> str:
        """Return the file name for the calibration file.
        """
        if self.sim:
            filename = "sim"
        else:
            filename = f"proto"
        filename += f"_{self.intent}_{self.chip}_"


def generate_calibration_file(calibration_type: CalibrationType, chip_name: str,
    mean: float, rms: float, version: int = 1) -> None:
    """Generate a calibration file for the given calibration type and chip name.
    """
    # This might be something along the lines of
    # sim_xpol3_enc-20_uniform_v001.h5
    # sim_xpol3_enc-20_gauss-p10_v001.h5
    file_name = f"sim_{chip_name}.h5"


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
