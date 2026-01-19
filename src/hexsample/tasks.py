# Copyright (C) 2025--2026 the hexsample team.
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
import pathlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from aptapy.hist import Histogram1d, Histogram2d
from aptapy.plotting import plt, setup_gca
from tqdm import tqdm

from . import rng
from .analysis import create_histogram
from .clustering import ClusteringNN
from .display import HexagonalGridDisplay
from .fileio import (
    DigiInputFileRectangular,
    ReconInputFile,
    ReconOutputFile,
    digi_input_file_class,
    digi_output_file_class,
    peek_readout_type,
)
from .hexagon import HexagonalLayout
from .logging_ import logger
from .mc import PhotonList
from .readout import (
    AbstractReadout,
    HexagonalReadoutCircular,
    HexagonalReadoutMode,
    HexagonalReadoutRectangular,
)
from .recon import ReconEvent
from .sensor import Sensor
from .source import Source

# Make room for the output data.
HEXSAMPLE_DATA = pathlib.Path.home() / "hexsampledata"
if not HEXSAMPLE_DATA.exists():
    logger.info(f"Creating data folder {HEXSAMPLE_DATA}...")
    pathlib.Path.mkdir(HEXSAMPLE_DATA)


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

    This is a small helper dataclass to help ensure consistency between the main task
    definition in this Python module and the command-line interface.
    """
    # source: Source = Source()
    # sensor: Sensor = Sensor()
    # readout = ? For this one we need to reconcile the thing with argparse...
    num_events: int = 10000
    output_file_path: str = HEXSAMPLE_DATA / "simulation.h5"
    random_seed: int = None


def simulate(
        source: Source,
        sensor: Sensor,
        readout: AbstractReadout,
        num_events: int = SimulationDefaults.num_events,
        output_file_path: str = SimulationDefaults.output_file_path,
        random_seed: int = SimulationDefaults.random_seed,
        # This will go away.
        header_kwargs: dict = None,
        ) -> str:
    """Run a simulation.

    .. warning::

       The last `header_kwargs` argument is a temporary workaround to allow passing
       some metadata to be stored in the output file header. This will go away once
       we have a proper mechanism to handle metadata.

    Arguments
    ----------
    source : Source
        The X-ray source.

    sensor : Sensor
        The sensor.

    readout : AbstractReadout
        The readout chip.

    num_events : int
        The number of events to simulate.

    output_file_path : str
        The path to the output file.

    random_seed : int
        The random seed to use.

    Returns
    -------
    str
        The path to the output file that the task has created.
    """
    name, args = current_call()
    logger.info(f"Running {__name__}.{name} with arguments {args}...")
    rng.initialize(seed=random_seed)
    photon_list = PhotonList(source, sensor, num_events)
    file_type = digi_output_file_class(readout)
    output_file = file_type(output_file_path)
    # This is just a momentary workaround until we write all the metadata in the
    # hdf5 file properly.
    if header_kwargs is not None:
        output_file.update_header(**header_kwargs)
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

    This is a small helper dataclass to help ensure consistency between the main task
    definition in this Python module and the command-line interface.
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
        header_kwargs: dict = None,
        ) -> str:
    """Run the reconstruction.

    .. warning::

       The last `header_kwargs` argument is a temporary workaround to allow passing
       some metadata to be stored in the output file header. This will go away once
       we have a proper mechanism to handle metadata.

    Arguments
    ----------
    input_file_path : str
        The path to the input file.

    suffix : str
        The suffix to append to the output file name.

    zero_sup_threshold : int
        The zero-suppression threshold.

    num_neighbors : int
        The number of neighbor pixels to be used for the clustering.

    pos_recon_algorithm : str
        The position reconstruction algorithm to use.

    eta_index : float
        The eta index to use.
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
    if header_kwargs is not None:
        output_file.update_header(**header_kwargs)
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


class DisplayDefaults:
    """Default parameters for the display task.

    This is a small helper dataclass to help ensure consistency between the main task
    definition in this Python module and the command-line interface.
    """


def display(input_file_path: str) -> None:
    """Display events from a digi file.

    Arguments
    ---------
    file_path : str
        The path to the digi file.
    """
    name, args = current_call()
    logger.info(f"Running {__name__}.{name} with arguments {args}...")
    input_file = DigiInputFileRectangular(input_file_path)
    header = input_file.header
    args = HexagonalLayout(header["layout"]), header["num_cols"], header["num_rows"],\
        header["pitch"], header["enc"], header["gain"]
    readout = HexagonalReadoutRectangular(*args)
    logger.info(f"Readout chip: {readout}")
    grid_display = HexagonalGridDisplay(readout)
    for event in input_file:
        print(event.ascii())
        grid_display.draw_digi_event(event, zero_sup_threshold=0)
        grid_display.show()
    input_file.close()


class QuickLookDefaults:
    """Default parameters for the quicklook task.

    This is a small helper dataclass to help ensure consistency between the main task
    definition in this Python module and the command-line interface.
    """


def quicklook(input_file_path: str) -> None:
    """Quicklook at events from a recon file.

    .. warning::
       This needs to be rebuilt from the ground up, but the intent is a good one, I think.

    Arguments
    ---------
    file_path : str
        The path to the input recon file.
    """
    name, args = current_call()
    logger.info(f"Running {__name__}.{name} with arguments {args}...")
    input_file = ReconInputFile(input_file_path)
    # Plotting the reconstructed energy and the true energy
    histo = create_histogram(input_file, "energy", mc=False)
    mc_histo = create_histogram(input_file, "energy", mc=True, binning=histo.bin_edges())
    plt.figure("Photons energy")
    histo.plot(label="Reconstructed")
    mc_histo.plot(label="MonteCarlo")
    plt.xlabel("Energy [eV]")
    plt.legend()

    # Plotting the reconstructed x and y position and the true position.
    plt.figure("Reconstructed photons position")
    binning = np.linspace(-5. * 0.1, 5. * 0.1, 100)
    x = input_file.column("posx")
    y = input_file.column("posy")
    histo = Histogram2d(binning, binning).fill(x, y)
    histo.plot()
    setup_gca(xlabel="x [cm]", ylabel="y [cm]")
    plt.figure("True photons position")
    x_mc = input_file.mc_column("absx")
    y_mc = input_file.mc_column("absy")
    histo_mc = Histogram2d(binning, binning).fill(x_mc, y_mc)
    histo_mc.plot()
    setup_gca(xlabel="x [cm]", ylabel="y [cm]")
    #Closing the file and showing the figures.
    plt.figure("x-direction resolution")
    binning = np.linspace((x-x_mc).min(), (x-x_mc).max(), 100)
    histx = Histogram1d(binning, xlabel=r"$x - x_{MC}$ [cm]").fill(x-x_mc)
    histx.plot()
    plt.figure("y-direction resolution")
    binning = np.linspace((y-y_mc).min(), (y-y_mc).max(), 100)
    histy = Histogram1d(binning, xlabel=r"$y - y_{MC}$ [cm]").fill(y-y_mc)
    histy.plot()

    input_file.close()
    plt.show()
