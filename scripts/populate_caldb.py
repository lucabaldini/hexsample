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

from hexsample.caldb import (
    CalDB,
    CalibrationType,
    GenerateCalibrationDefaults,
    generate_calibration_file
)

ENC_VALS = [20, 30, 40, 50, 75, 100]
ENC_RMS = [0, 10, 20]

NOISE_VALS = [20, 30, 40, 50, 75, 100]
NOISE_RMS = [0, 10, 20]

PEDESTAL_VALS = [0, 1000]
PEDESTAL_RMS = [0]

GAIN_VALS = [1]
GAIN_RMS = [0, 10, 20]

default_kwargs = dict(
    chip_name=GenerateCalibrationDefaults.chip_name,
    version=GenerateCalibrationDefaults.version,
    random_seed=GenerateCalibrationDefaults.random_seed,
)

root_dir = CalDB.DEFAULT_DIR


def generate_files(
        calibration_type: str,
        mean_vals: list,
        rms_vals: list
        ) -> None:
    for mean in mean_vals:
        for rms in rms_vals:
            generate_calibration_file(
                calibration_type=calibration_type,
                mean=mean,
                rms=rms,
                output_dir=root_dir / calibration_type,
                **default_kwargs
            )


if __name__ == "__main__":
    # generate_files(CalibrationType.ENC,
    #                ENC_VALS,
    #                ENC_RMS)
    # generate_files(CalibrationType.NOISE,
    #                NOISE_VALS,
    #                NOISE_RMS)
    generate_files(CalibrationType.PEDESTAL,
                   PEDESTAL_VALS,
                   PEDESTAL_RMS)
    # generate_files(CalibrationType.GAIN,
    #                GAIN_VALS,
    #                GAIN_RMS)
