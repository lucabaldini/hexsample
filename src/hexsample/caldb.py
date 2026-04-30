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
from enum import Enum
from typing import Tuple

from .calibration import CalibrationMatrix


class CalibrationType(str, Enum):

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

    DEFAULT_DIR = pathlib.Path(__file__).parent.parent.parent / "caldb"

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
