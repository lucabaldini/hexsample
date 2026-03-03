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

from . import legacy, tasks
from .calibration import CalibrationMatrixGain
from .readout import ReadoutProxy
from .sensor import Sensor
from .source import Source


def simulate(**kwargs) -> str:
    """Run a simulation.
    """
    defaults = tasks.SimulationDefaults
    source = Source.from_filtered_kwargs(**kwargs)
    sensor = Sensor.from_filtered_kwargs(**kwargs)
    # If a gain response file is provided, load the gain matrix and update the kwargs
    if kwargs.get("map_gain_file") is not None:
        gain_file = CalibrationMatrixGain.from_hdf5(kwargs.get("map_gain_file"))
        kwargs.update({"gain": gain_file.matrix})
    readout = ReadoutProxy.from_filtered_kwargs(**kwargs)
    num_events = kwargs.get("num_events", defaults.num_events)
    output_file_path = kwargs.get("output_file", defaults.output_file_path)
    random_seed = kwargs.get("random_seed", defaults.random_seed)
    args = source, sensor, readout, num_events, output_file_path, random_seed
    return tasks.simulate(*args, kwargs)


def reconstruct(**kwargs) -> str:
    """Run a reconstruction.
    """
    defaults = tasks.ReconstructionDefaults
    input_file_path = kwargs["input_file"]
    suffix = kwargs.get("suffix", defaults.suffix)
    zero_sup_threshold = kwargs.get("zero_sup_threshold", defaults.zero_sup_threshold)
    num_neighbors = kwargs.get("num_neighbors", defaults.num_neighbors)
    max_neighbors = kwargs.get("max_neighbors", defaults.max_neighbors)
    pos_recon_algorithm = kwargs.get("pos_recon_algorithm", defaults.pos_recon_algorithm)
    map_gain_file = kwargs.get("map_gain_file")
    if map_gain_file is not None:
        gain_map = CalibrationMatrixGain.from_hdf5(map_gain_file).matrix
    else:
        gain_map = None
    eta_2pix_rad = kwargs.get("eta_2pix_rad", defaults.eta_2pix_rad)
    eta_2pix_pivot = kwargs.get("eta_2pix_pivot", defaults.eta_2pix_pivot)
    eta_3pix_rad0 = kwargs.get("eta_3pix_rad0", defaults.eta_3pix_rad0)
    eta_3pix_rad1 = kwargs.get("eta_3pix_rad1", defaults.eta_3pix_rad1)
    eta_3pix_rad_pivot = kwargs.get("eta_3pix_rad_pivot", defaults.eta_3pix_rad_pivot)
    eta_3pix_theta0 = kwargs.get("eta_3pix_theta0", defaults.eta_3pix_theta0)
    args = input_file_path, suffix, zero_sup_threshold, num_neighbors, max_neighbors, \
           pos_recon_algorithm, gain_map, eta_2pix_rad, eta_2pix_pivot, \
           eta_3pix_rad0, eta_3pix_rad1, eta_3pix_rad_pivot, eta_3pix_theta0
    return tasks.reconstruct(*args, kwargs)


def calibrate(**kwargs) -> None:
    """Calibrate the gain and noise of the chip.
    """
    input_file_path = kwargs["input_file"]
    energy = kwargs["energy"]
    num_events = kwargs.get("num_events", tasks.CalibrationDefaults.num_events)
    zero_sup_threshold = kwargs.get("zero_sup_threshold",
                                    tasks.CalibrationDefaults.zero_sup_threshold)
    default_gain = kwargs.get("default_gain", tasks.CalibrationDefaults.default_gain)
    default_noise = kwargs.get("default_noise", tasks.CalibrationDefaults.default_noise)
    return tasks.calibrate(input_file_path, energy, num_events,zero_sup_threshold, default_gain,
                           default_noise)


def display(**kwargs) -> None:
    """Display events from a digi or recon file.
    """
    input_file_path = kwargs["input_file"]
    zero_sup_threshold = kwargs.get("zero_sup_threshold", tasks.DisplayDefaults.zero_sup_threshold)
    return tasks.display(input_file_path, zero_sup_threshold)


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
