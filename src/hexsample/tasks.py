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

import inspect
from dataclasses import dataclass
from typing import Tuple

from tqdm import tqdm

from . import rng
from .clustering import ClusteringNN
from .fileio import digi_input_file_class, digi_output_file_class, peek_readout_type, ReconOutputFile
from .hexagon import HexagonalLayout
from .logging_ import logger
from .mc import PhotonList
from .readout import AbstractReadout, HexagonalReadoutCircular, HexagonalReadoutMode, HexagonalReadoutRectangular
from .recon import ReconEvent
from .sensor import Sensor
from .source import Source


def current_call() -> Tuple[str, dict]:
    """Return the name and arguments of the current function call.
    """
    frame = inspect.currentframe().f_back
    func = frame.f_code.co_name
    sig = inspect.signature(frame.f_globals[func])
    bound = sig.bind(**frame.f_locals)
    bound.apply_defaults()
    return func, bound.arguments


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
        # This will go away.
        kwargs: dict = None,
        ) -> str:
    """Run a simulation.
    """
    name, args = current_call()
    logger.info(f"Running {__name__}.{name} with arguments {args}...")
    rng.initialize(seed=random_seed)
    photon_list = PhotonList(source, sensor, num_events)
    file_type = digi_output_file_class(readout)
    output_file = file_type(output_file_path)
    # This is just a momentary workaround until we write all the metadata in the
    # hdf5 file properly.
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
    zero_sup_threshold: int = 0
    num_neighbors: int = 2
    pos_recon_algorithm: str = "centroid"
    eta_index: float = 0.27


def reconstruct(
        input_file_path: str,
        suffix: str = ReconstructionDefaults.suffix,
        zero_sup_threshold: int = ReconstructionDefaults.zero_sup_threshold,
        num_neighbors: int = ReconstructionDefaults.num_neighbors,
        pos_recon_algorithm: str = ReconstructionDefaults.pos_recon_algorithm,
        eta_index: float = ReconstructionDefaults.eta_index,
        # This will go away.
        **kwargs,
        ) -> str:
    """Run the reconstruction.
    """
    name, args = current_call()
    logger.info(f"Running {__name__}.{name} with arguments {args}...")
    # Note we cast the input file to string, in case it happens to be a pathlib.Path object.
    input_file_path = str(input_file_path)
    if not input_file_path.endswith(".h5"):
        raise RuntimeError("Input file {input_file_path} does not look like a HDF5 file")

    # It is necessary to extract the reaodut type because every readout type
    # corresponds to a different DigiEvent type.
    readout_mode = peek_readout_type(input_file_path)
    # And we should get rid of all this crap when we store the readout type and all the
    # relevant metadata in the hdf5 file in a sensible way.
    file_type = digi_input_file_class(readout_mode)
    input_file = file_type(input_file_path)
    header = input_file.header
    args = HexagonalLayout(header["layout"]), header["num_cols"], header["num_rows"],\
        header["pitch"], header["enc"], header["gain"]
    if readout_mode is HexagonalReadoutMode.RECTANGULAR:
        readout = HexagonalReadoutRectangular(*args, padding=header["padding"])
    elif readout_mode is HexagonalReadoutMode.CIRCULAR:
        readout = HexagonalReadoutCircular(*args)
    else:
        raise RuntimeError(f"Unsupported readout mode: {readout_mode}")
    logger.info(f"Readout chip: {readout}")

    # Run the actual reconstruction.
    clustering = ClusteringNN(readout, zero_sup_threshold, num_neighbors)
    output_file_path = input_file_path.replace(".h5", f"_{suffix}.h5")
    output_file = ReconOutputFile(output_file_path)
    output_file.update_header(**kwargs)
    output_file.update_digi_header(**input_file.header)
    for i, event in tqdm(enumerate(input_file)):
        cluster = clustering.run(event)
        if num_neighbors == 0 or cluster.size() == num_neighbors:
            # Need to pass the recon method and other stuff as argument to ReconEvent
            args = event.trigger_id, event.timestamp(), event.livetime, cluster
            recon_event = ReconEvent(*args, pos_recon_algorithm, readout.pitch, eta_index)
            mc_event = input_file.mc_event(i)
            output_file.add_row(recon_event, mc_event)
    output_file.flush()
    input_file.close()
    output_file.close()
    return output_file_path