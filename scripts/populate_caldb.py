#!/usr/bin/env python
#
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

"""Script to populate the calibration database with some calibration files.
"""

import numpy as np

from hexsample.caldb import CalDB, CalibrationType
from hexsample.tasks import SynthesizeCalibrationDefaults, synthesize_calibration_file

DEFAULT_GAIN = 0.1
RMS_VALS = (0, 10)
ENC_VALS = np.array((20., 30., 40., 50., 75., 100.))
PEDESTAL_VALS = (0, 1000)

default_kwargs = dict(
    chip_name=SynthesizeCalibrationDefaults.chip_name,
    version=SynthesizeCalibrationDefaults.version,
    random_seed=SynthesizeCalibrationDefaults.random_seed,
)

root_dir = CalDB.DEFAULT_DIR

def generate_files(
        calibration_type: str,
        mean_vals: list,
        rms_vals: list
        ) -> None:
    output_dir = root_dir / calibration_type
    output_dir.mkdir(parents=True, exist_ok=True)
    for mean in mean_vals:
        for rms in rms_vals:
            synthesize_calibration_file(
                calibration_type=calibration_type,
                mean=mean,
                percent_rms=rms,
                output_dir=output_dir,
                **default_kwargs
            )


def populate_caldb() -> None:
    generate_files(CalibrationType.ENC, ENC_VALS, RMS_VALS)
    generate_files(CalibrationType.NOISE, ENC_VALS * DEFAULT_GAIN, RMS_VALS)
    generate_files(CalibrationType.GAIN, [DEFAULT_GAIN], RMS_VALS)
    generate_files(CalibrationType.PEDESTAL, PEDESTAL_VALS, RMS_VALS)


if __name__ == "__main__":
    populate_caldb()
