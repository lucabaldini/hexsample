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

from tqdm import tqdm

from . import rng
from .logging_ import logger
from .mc import PhotonList
from .readout import AbstractReadout
from .sensor import Sensor
from .source import Source


def simulate(
        source: Source,
        sensor: Sensor,
        readout: AbstractReadout,
        num_events: int,
        output_file_path: str,
        random_seed: int = None
        ) -> str:
    """Run a simulation.
    """
    rng.initialize(seed=random_seed)
    logger.info("Setting up the simulation...")
    logger.info(source)
    logger.info(sensor)
    logger.info(readout)
    photon_list = PhotonList(source, sensor, num_events)
    file_type = readout.output_file_class()
    output_file = file_type(output_file_path)
    kwargs = {}
    # Need to all all metadata here!
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


def reconstruct(
        input_file_path: str,
        suffic: str = "recon",
        ) -> str:
    """Run the reconstruction.
    """
    pass