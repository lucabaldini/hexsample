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

# from tqdm import tqdm

# from hexsample import rng
# from hexsample.fileio import digioutput_class
# from hexsample.hexagon import HexagonalGrid, HexagonalLayout
# from hexsample.logging_ import logger
# from hexsample.mc import PhotonList
# from hexsample.readout import HexagonalReadoutMode, readout_chip
# from hexsample.roi import Padding
# from hexsample.sensor import Material, Sensor
# from hexsample.source import GaussianBeam, HexagonalBeam, LineForest, Source, TriangularBeam

from hexsample.readout import AbstractReadout
from hexsample.sensor import Sensor
from hexsample.source import Source


def simulate(source: Source, sensor: Sensor, readout: AbstractReadout) -> str:
    """Run a simulation.
    """
    print("Running a simulation...")
    # rng.initialize(seed=kwargs["seed"])
    print(source)
    print(sensor)
    print(readout)
    # photon_list = PhotonList(source, sensor, kwargs["numevents"])
    # logger.info(f"Readout chip: {readout}")
    # output_file_path = kwargs.get("outfile")
    # output_file = digioutput_class(readout_mode)(output_file_path)
    # output_file.update_header(**kwargs)
    # logger.info("Starting the event loop...")
    # for mc_event in tqdm(photon_list):
    #     x, y = mc_event.propagate(sensor.trans_diffusion_sigma)
    #     digi_event = readout.read(mc_event.timestamp, x, y, *readout_args)
    #     output_file.add_row(digi_event, mc_event)
    # logger.info("Done!")
    # output_file.flush()
    # output_file.close()
    # return output_file_path