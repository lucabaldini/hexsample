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
from typing import Type

from .calibration import CalibrationBase, CalibrationMatrix, CalibrationType, PositionCalibrationData


class CalDB:

    """Simple calibration database implementation.
    """

    ROOT_DIR = pathlib.Path(__file__).parent.parent.parent / "caldb"

    @classmethod
    def _open(cls, calibration_type: CalibrationType, designator: str,
              calibration_class: Type[CalibrationBase] = CalibrationMatrix) -> CalibrationBase:
        """Open the calibration file for the given designation and intent.
        """
        if pathlib.Path(designator).is_file():
            return calibration_class.from_hdf5(designator)
        file_path = cls.ROOT_DIR / calibration_type.value / f"{designator}.h5"
        return calibration_class.from_hdf5(file_path)

    @classmethod
    def open_enc(cls, designator: str) -> CalibrationMatrix:
        """Open the ENC calibration file for the given designation.
        """
        return cls._open(CalibrationType.ENC, designator)

    @classmethod
    def open_pedestal(cls, designator: str) -> CalibrationMatrix:
        """Open the pedestal calibration file for the given designation.
        """
        return cls._open(CalibrationType.PEDESTAL, designator)

    @classmethod
    def open_noise(cls, designator: str) -> CalibrationMatrix:
        """Open the noise calibration file for the given designation.
        """
        return cls._open(CalibrationType.NOISE, designator)

    @classmethod
    def open_gain(cls, designator: str) -> CalibrationMatrix:
        """Open the gain calibration file for the given designation.
        """
        return cls._open(CalibrationType.GAIN, designator)

    @classmethod
    def open_equalization(cls, designator: str) -> CalibrationMatrix:
        """Open the equalization calibration file for the given designation.
        """
        return cls._open(CalibrationType.EQUALIZATION, designator)

    @classmethod
    def open_position(cls, designator: str) -> PositionCalibrationData:
        """Open the MLE calibration file for the given designation.
        """
        return cls._open(CalibrationType.POSITION, designator, PositionCalibrationData)
