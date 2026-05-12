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

"""Pipeline facilities.
"""


from typing import Tuple

from . import legacy, tasks
from .readout import ReadoutProxy
from .sensor import Sensor
from .source import Source


def simulate(**kwargs) -> str:
    """Run a simulation.
    """
    defaults = tasks.SimulationDefaults
    source = Source.from_filtered_kwargs(**kwargs)
    sensor = Sensor.from_filtered_kwargs(**kwargs)
    readout = ReadoutProxy.from_filtered_kwargs(**kwargs)
    num_events = kwargs.get("num_events", defaults.num_events)
    output_file_path = kwargs.get("output_file", defaults.output_file_path)
    random_seed = kwargs.get("random_seed", defaults.random_seed)
    args = source, sensor, readout, num_events, output_file_path, random_seed
    return tasks.simulate(*args, kwargs)


def reconstruct(**kwargs) -> str:
    """Run a reconstruction.
    """
    defaults = tasks.ReconstructionDefaults()
    input_file_path = kwargs["input_file"]
    noise_matrix = kwargs["noise"]
    pedestal_matrix = kwargs["pedestal"]
    equalization_matrix = kwargs["equalization"]
    suffix = kwargs.get("suffix", defaults.suffix)
    zero_sup_threshold = kwargs.get("zero_sup_threshold", defaults.zero_sup_threshold)
    num_neighbors = kwargs.get("num_neighbors", defaults.num_neighbors)
    max_neighbors = kwargs.get("max_neighbors", defaults.max_neighbors)
    pos_recon_algorithm = kwargs.get("pos_recon_algorithm", defaults.pos_recon_algorithm)
    eta_2pix_rad_sigma = kwargs.get("eta_2pix_rad_sigma", defaults.eta_2pix_rad_sigma)
    eta_2pix_rad_pivot = kwargs.get("eta_2pix_rad_pivot", defaults.eta_2pix_rad_pivot)
    eta_3pix_rad_offset = kwargs.get("eta_3pix_rad_offset", defaults.eta_3pix_rad_offset)
    eta_3pix_rad_sigma = kwargs.get("eta_3pix_rad_sigma", defaults.eta_3pix_rad_sigma)
    eta_3pix_rad_pivot = kwargs.get("eta_3pix_rad_pivot", defaults.eta_3pix_rad_pivot)
    eta_3pix_theta_sigma = kwargs.get("eta_3pix_theta_sigma", defaults.eta_3pix_theta_sigma)
    recon_args = (eta_2pix_rad_sigma, eta_2pix_rad_pivot, eta_3pix_rad_offset, eta_3pix_rad_sigma,
                  eta_3pix_rad_pivot, eta_3pix_theta_sigma)
    args = input_file_path, noise_matrix, pedestal_matrix, equalization_matrix, suffix, \
            zero_sup_threshold,num_neighbors, max_neighbors, pos_recon_algorithm, *recon_args
    return tasks.reconstruct(*args, kwargs)


def calibspec(**kwargs) -> str:
    """Create a probability density function from a reconstructed spectrum
    to use in the gain calibration.
    """
    input_file_path = kwargs["input_file"]
    return tasks.calibspec(input_file_path)


def calibrate_eta(**kwargs) -> None:
    """Calibrate the eta function using the events from a digi file.
    """
    input_file_path = kwargs["input_file"]
    equalization_matrix = kwargs["equalization"]
    noise_matrix = kwargs["noise"]
    pedestal_matrix = kwargs["pedestal"]
    num_bins = kwargs.get("num_bins", tasks.CalibrationEtaDefaults.num_bins)
    zero_sup_threshold = kwargs.get("zero_sup_threshold",
                                    tasks.CalibrationEtaDefaults.zero_sup_threshold)
    args = input_file_path, noise_matrix, pedestal_matrix, equalization_matrix, num_bins, \
            zero_sup_threshold
    return tasks.calibrate_eta(*args)


def calibrate_noise(**kwargs) -> str:
    """Calibrate the noise of the chip.
    """
    input_file_path = kwargs["input_file"]
    return tasks.calibrate_noise(input_file_path)


def calibrate_dark(**kwargs) -> Tuple[str, str]:
    """Calibrate the dark of the chip.
    """
    input_file_path = kwargs["input_file"]
    algorithm = kwargs["algorithm"]
    has_source = kwargs["has_source"]
    batch_size = kwargs["batch_size"]
    args = input_file_path, algorithm, has_source, batch_size
    return tasks.calibrate_dark(*args)


def calibrate_enc(**kwargs) -> str:
    """Calibrate the ENC of the chip.
    """
    noise_matrix = kwargs["noise"]
    gain_matrix = kwargs["gain"]
    output_dir = kwargs.get("output_dir", tasks.CalibrationEncDefaults.output_dir)
    args = noise_matrix, gain_matrix, output_dir
    return tasks.calibrate_enc(*args)


def calibrate_equalization(**kwargs) -> str:
    """Calibrate the equalization of the chip.
    """
    input_file_path = kwargs["input_file"]
    pdf = kwargs["pdf"]
    noise_matrix = kwargs["noise"]
    pedestal_matrix = kwargs["pedestal"]
    size = kwargs.get("size", tasks.CalibrationEqualizationDefaults.size)
    zero_sup_threshold = kwargs.get("zero_sup_threshold",
                                    tasks.CalibrationEqualizationDefaults.zero_sup_threshold)
    args = input_file_path, pdf, noise_matrix, pedestal_matrix, size, zero_sup_threshold
    return tasks.calibrate_equalization(*args)


def calibrate_gain(**kwargs) -> str:
    """Calibrate the gain of the chip.
    """
    equalization_matrix = kwargs["equalization"]
    material_symbol = kwargs.get("material_symbol", tasks.CalibrationGainDefaults.material_symbol)
    output_dir = kwargs.get("output_dir", tasks.CalibrationGainDefaults.output_dir)
    args = equalization_matrix, material_symbol, output_dir
    return tasks.calibrate_gain(*args)


def synthesize_calibration_file(**kwargs) -> str:
    """Generate a calibration file of a given type.
    """
    defaults = tasks.SynthesizeCalibrationDefaults
    calibration_type = kwargs["calibration_type"]
    mean = kwargs["mean"]
    rms = kwargs.get("percent_rms", defaults.percent_rms)
    chip_name = kwargs.get("chip_name", defaults.chip_name)
    output_dir = kwargs.get("output_dir", defaults.output_dir)
    version = kwargs.get("version", defaults.version)
    random_seed = kwargs.get("random_seed", defaults.random_seed)
    args = calibration_type, mean, rms, chip_name, output_dir, version, random_seed
    return tasks.synthesize_calibration_file(*args)


def display(**kwargs) -> None:
    """Display events from a digi or recon file.
    """
    input_file_path = kwargs["input_file"]
    equalization_matrix = kwargs["equalization"]
    noise_matrix = kwargs["noise"]
    pedestal_matrix = kwargs["pedestal"]
    args = input_file_path, noise_matrix, pedestal_matrix, equalization_matrix
    return tasks.display(*args)


def quicklook(**kwargs) -> None:
    """Quicklook at events from a recon file.
    """
    input_file_path = kwargs["input_file"]
    return tasks.quicklook(input_file_path)


def mdat3_to_digi(**kwargs) -> None:
    """Convert a .mdat3 file to a HDF5 digi file.
    """
    file_path = kwargs["input_file"]
    num_events = kwargs.get("num_events")
    return legacy.mdat3_to_digi(file_path, num_events)


def calibview(**kwargs) -> None:
    """Inspect a calibration matrix.
    """
    matrix = kwargs["matrix"]
    mc_matrix = kwargs.get("mc_matrix")
    min_hits = kwargs.get("min_hits", tasks.CalibviewDefaults.min_hits)
    rel_error = kwargs.get("rel_error", tasks.CalibviewDefaults.rel_error)
    lower_quantile = kwargs.get("lower_quantile", tasks.CalibviewDefaults.lower_quantile)
    upper_quantile = kwargs.get("upper_quantile", tasks.CalibviewDefaults.upper_quantile)
    args = matrix, mc_matrix, min_hits, rel_error, lower_quantile, upper_quantile
    return tasks.calibview(*args)
