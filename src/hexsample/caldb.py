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

from . import rng
from .calibration import CalibrationMatrix
from .tasks import HEXSAMPLE_DATA
from .xpol import XPOL, XPOL_CHIP_DICT


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


@dataclass(frozen=True)
class GenerateCalibrationDefaults:

    """Default values for the generate_calibration_file task.
    """

    rms: int = 0
    output_dir: str = HEXSAMPLE_DATA
    chip_name: str = XPOL.XPOL3.value
    version: int = 1
    random_seed: int = None


def generate_calibration_file(
        calibration_type: CalibrationType,
        mean: float,
        rms: int = GenerateCalibrationDefaults.rms,
        chip_name: str = GenerateCalibrationDefaults.chip_name,
        output_dir: str = GenerateCalibrationDefaults.output_dir,
        version: int = GenerateCalibrationDefaults.version,
        random_seed: int = GenerateCalibrationDefaults.random_seed
        ) -> str:
    """Generate a calibration file for the given calibration type and chip name.
    """
    # Initialize the random number generator with the given seed
    rng.initialize(seed=random_seed)
    # Check the validity of the input chip name
    if chip_name not in XPOL.values():
        raise ValueError(f"Unsupported chip: {chip_name}. Choose from {list(XPOL.values())}")
    # Generate the file name
    file_name = f"sim_{chip_name}_{calibration_type}-{mean:g}".replace(".", "p")
    # Append the RMS information to the file name
    if rms > 0:
        file_name += f"_gauss-p{rms:02d}".replace(".", "p")
    elif rms == 0:
        file_name += "_uniform"
    else:
        raise ValueError("RMS must be non-negative")
    # Append the version number to the file name
    file_name += f"_v{version:03d}.h5"
    # Generate the calibration matrix with the appropriate size and values
    num_cols, num_rows = XPOL_CHIP_DICT[chip_name]
    calibration_matrix = CalibrationMatrix(num_cols, num_rows)
    calibration_matrix.values = rng.generator.normal(mean, scale=mean*rms/100,
                                                     size=(num_rows, num_cols))
    # Save the calibration matrix to the output directory
    output_path = pathlib.Path(output_dir) / file_name
    calibration_matrix.to_hdf5(output_path, calibration_type, True)
    return str(output_path)
