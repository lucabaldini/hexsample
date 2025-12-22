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

from hexsample import rng
from hexsample.fileio import digioutput_class
from hexsample.hexagon import HexagonalGrid, HexagonalLayout
from hexsample.logging_ import logger
from hexsample.mc import PhotonList
from hexsample.readout import HexagonalReadoutMode, readout_chip
from hexsample.roi import Padding
from hexsample.sensor import Material, Sensor
from hexsample.source import GaussianBeam, HexagonalBeam, LineForest, Source, TriangularBeam


def simulate(**kwargs):
    """Run a simulation.
    """
    # pylint: disable=too-many-locals, invalid-name
    rng.initialize(seed=kwargs["seed"])
    spectrum = LineForest(kwargs["srcelement"], kwargs["srclevel"])
    grid_args = HexagonalLayout(kwargs["layout"]), kwargs["numcolumns"], kwargs["numrows"],\
        kwargs["pitch"]
    if kwargs["beamshape"] == "gaussian":
        beam = GaussianBeam(kwargs["srcposx"], kwargs["srcposy"], kwargs["srcsigma"])
    elif kwargs["beamshape"] == "triangular":
        grid = HexagonalGrid(*grid_args)
        target_col, target_row = grid.world_to_pixel(kwargs["srcposx"], kwargs["srcposy"])
        center, v0, v1 = grid.find_vertices(target_col, target_row, kwargs["trngindex"])
        beam = TriangularBeam(*center, tuple(v0), tuple(v1))
    elif kwargs["beamshape"] == "hexagonal":
        grid = HexagonalGrid(*grid_args)
        target_col, target_row = grid.world_to_pixel(kwargs["srcposx"], kwargs["srcposy"])
        center, v0, v1 = grid.find_vertices(target_col, target_row)
        beam = HexagonalBeam(*center, tuple(v0), tuple(v1))
    else:
        raise RuntimeError
    source = Source(spectrum, beam)
    material = Material(kwargs["actmedium"], kwargs["fano"])
    sensor = Sensor(material, kwargs["thickness"], kwargs["transdiffsigma"])
    photon_list = PhotonList(source, sensor, kwargs["numevents"])
    readout_mode = HexagonalReadoutMode(kwargs["readoutmode"])
    # Is there any nicer way to do this? See https://github.com/lucabaldini/hexsample/issues/51
    if readout_mode is HexagonalReadoutMode.SPARSE:
        readout_args = kwargs["trgthreshold"], kwargs["zsupthreshold"], kwargs["offset"]
    elif readout_mode is HexagonalReadoutMode.RECTANGULAR:
        padding = Padding(*kwargs["padding"])
        readout_args = kwargs["trgthreshold"], padding, kwargs["zsupthreshold"], kwargs["offset"]
    elif readout_mode is HexagonalReadoutMode.CIRCULAR:
        readout_args = kwargs["trgthreshold"], kwargs["zsupthreshold"], kwargs["offset"]
    else:
        raise RuntimeError
    args = HexagonalLayout(kwargs["layout"]), kwargs["numcolumns"], kwargs["numrows"],\
        kwargs["pitch"], kwargs["noise"], kwargs["gain"]
    readout = readout_chip(readout_mode, *args)
    logger.info(f"Readout chip: {readout}")
    output_file_path = kwargs.get("outfile")
    output_file = digioutput_class(readout_mode)(output_file_path)
    output_file.update_header(**kwargs)
    logger.info("Starting the event loop...")
    for mc_event in tqdm(photon_list):
        x, y = mc_event.propagate(sensor.trans_diffusion_sigma)
        digi_event = readout.read(mc_event.timestamp, x, y, *readout_args)
        output_file.add_row(digi_event, mc_event)
    logger.info("Done!")
    output_file.flush()
    output_file.close()
    return output_file_path