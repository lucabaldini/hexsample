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

"""Analyze the noise to find the best the zero suppression threshold.
"""

import argparse
import pathlib

import numpy as np
from aptapy.plotting import plt

from hexsample.fileio import digi_input_file_class, peek_readout_type
from hexsample.pipeline import reconstruct, resolution
from hexsample.readout import HexagonalReadoutMode

ARGPARSER = argparse.ArgumentParser()
ARGPARSER.add_argument("input_file", type=str, help="path to the input file")
ARGPARSER.add_argument("--num_neighbors", type=int, default=2,
                    help="number of neighbors for the clustering (default: 2)")
ARGPARSER.add_argument("--pos_recon_algorithm", type=str, default="eta",
                    help="position reconstruction algorithm (default: eta)")


def analyze_noise(**kwargs):
    """ Analyze the effect of the noise and zero suppression threshold on the reconstucted position
    resolution, by calculating the bias and the standard deviation of the position residuals
    distribution for different values of the threshold.
    """
    # Open the file to get the header information
    input_file_path = str(kwargs["input_file"])
    if not input_file_path.endswith(".h5"):
        raise RuntimeError(f"Input file {input_file_path} does not look like a HDF5 file")
    readout_mode = peek_readout_type(input_file_path)
    if readout_mode is not HexagonalReadoutMode.CIRCULAR:
        raise RuntimeError("Only CIRCULAR readout is supported.")
    file_type = digi_input_file_class(readout_mode)
    input_file = file_type(input_file_path)
    header = input_file.header
    input_file.close()

    enc = header["enc"]
    # Scan different zero-suppression thresholds
    n_files = 10
    zero_sup_threshold_grid = np.linspace(0, 5 * enc, n_files, dtype=int)
    results = np.zeros((n_files, 3))
    default_folder = pathlib.Path.home() / "hexsampledata"
    for i, zero_sup_threshold in enumerate(zero_sup_threshold_grid):
        # Define the arguments for the reconstruction
        recon_kwargs = dict(
            input_file=input_file_path,
            suffix=(f"recon_enc{enc}_thr{zero_sup_threshold}_"
            f"{kwargs['pos_recon_algorithm']}_nn{kwargs['num_neighbors']}"),
            zero_sup_threshold=zero_sup_threshold,
            num_neighbors=kwargs["num_neighbors"],
            pos_recon_algorithm=kwargs["pos_recon_algorithm"],
        )
        # Check if reconstructed files already exist
        file_name = default_folder / (
            f"{pathlib.Path(recon_kwargs['input_file']).stem}_"
            f"{recon_kwargs['suffix']}.h5")
        if not file_name.with_suffix(".h5").exists():
            output_file_path = reconstruct(**recon_kwargs)
        else:
            output_file_path = str(file_name.with_suffix(".h5"))
        # Analyze the resolution of the reconstructed file
        res_kwargs = dict(
            input_file=output_file_path,
        )
        results[i] = resolution(**res_kwargs)

    plt.figure("Position resolution vs zero-suppression threshold")
    plt.plot(zero_sup_threshold_grid / enc, results[:, 0], ".k", label="Mean")
    plt.plot(zero_sup_threshold_grid / enc, results[:, 1], ".r", label="Stddev")
    # plt.plot(zero_sup_threshold_grid / enc, results[:, 2], ".b", label="FWHM")
    plt.xlabel("Zero-suppression threshold  / enc")
    plt.ylabel("Position resolution [pitch]")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    analyze_noise(**vars(ARGPARSER.parse_args()))

