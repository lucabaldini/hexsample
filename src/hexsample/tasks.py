# Copyright (C) 2025 the hexsample team.
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

"""Basic simulation, reconstruction and analysis tasks.
"""

from dataclasses import dataclass

from tqdm import tqdm

from . import rng
from .logging_ import logger
from .mc import PhotonList
from .readout import AbstractReadout
from .sensor import Sensor
from .source import Source


@dataclass(frozen=True)
class SimulationDefaults:
    """Default parameters for the simulation task.
    """
    num_events: int = 10000
    output_file_path: str = "simulation_output.h5"
    random_seed: int = None


def simulate(
        source: Source,
        sensor: Sensor,
        readout: AbstractReadout,
        num_events: int = SimulationDefaults.num_events,
        output_file_path: str = SimulationDefaults.output_file_path,
        random_seed: int = SimulationDefaults.random_seed,
        kwargs: dict = None,
        ) -> str:
    """Run a simulation.
    """
    rng.initialize(seed=random_seed)
    logger.info("Setting up the simulation...")
    logger.info(source)
    logger.info(sensor)
    logger.info(readout)
    photon_list = PhotonList(source, sensor, num_events)
    # Change this back to a function in the fileio module.
    file_type = readout.output_file_class()
    output_file = file_type(output_file_path)
    if kwargs is not None:
        output_file.update_header(**kwargs)
    logger.info("Starting the event loop...")
    for mc_event in tqdm(photon_list):
        x, y = mc_event.propagate(sensor.diffusion_sigma)
        digi_event = readout.read(mc_event.timestamp, x, y)
        output_file.add_row(digi_event, mc_event)
    logger.info("Done!")
    output_file.flush()
    output_file.close()
    return output_file_path


@dataclass(frozen=True)
class ReconstructionDefaults:
    """Default parameters for the reconstruction task.
    """
    suffix: str = "recon"


def reconstruct(
        input_file_path: str,
        suffix: str = ReconstructionDefaults.suffix,
        ) -> str:
    """Run the reconstruction.
    """
    pass