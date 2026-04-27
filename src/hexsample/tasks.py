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
from typing import Optional, Tuple, Union

import numpy as np
from aptapy.hist import Histogram1d, Histogram2d
from aptapy.plotting import plt, setup_gca
from tqdm import tqdm

from . import rng
from .analysis import create_histogram
from .calibration import (
    CalibrateDark,
    CalibrateGain,
    CalibrateNoise,
    CalibrationMatrix,
    angle,
    calibrate_dr_2pix,
    calibrate_dr_3pix,
    calibrate_theta_3pix,
    distance,
)
from .clustering import ClusteringNN
from .display import EventDisplay
from .fileio import (
    DigiInputFileBase,
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
    HexagonalReadoutBase,
    HexagonalReadoutCircular,
    HexagonalReadoutMode,
    HexagonalReadoutRectangular,
)
from .recon import ReconEvent
from .sensor import Sensor
from .source import DiskBeam, Line, Source

# Make room for the output data.
HEXSAMPLE_DATA = pathlib.Path.home() / "hexsampledata"
if not HEXSAMPLE_DATA.exists():
    logger.info(f"Creating data folder {HEXSAMPLE_DATA}...")
    pathlib.Path.mkdir(HEXSAMPLE_DATA)


def current_call(num_backward_steps: int = 2) -> Tuple[str, dict]:
    """Return the name and arguments of the current function call.

    Arguments
    ---------
    num_backward_steps : int
        The number of steps to go back in the call stack to find the function call.
    """
    frame = inspect.currentframe()
    for _ in range(num_backward_steps):
        frame = frame.f_back
    func = frame.f_code.co_name
    sig = inspect.signature(frame.f_globals[func])
    bound = sig.bind(**frame.f_locals)
    bound.apply_defaults()
    return func, bound.arguments


def open_file(input_file_path: Union[str, pathlib.Path]) -> Tuple[DigiInputFileBase, dict, str]:
    """Open a digi file and extract the header and the readout type.
    """
    name, args = current_call()
    logger.info(f"Running {__name__}.{name} with arguments {args}...")
    input_file_path = str(input_file_path)
    if not input_file_path.endswith(".h5"):
        raise RuntimeError(f"Input file {input_file_path} does not look like a HDF5 file")
    readout_mode = peek_readout_type(input_file_path)
    file_type = digi_input_file_class(readout_mode)
    input_file = file_type(input_file_path)
    header = input_file.header
    return input_file, header, readout_mode


def create_readout(readout_mode: HexagonalReadoutMode, header: dict, *args
                   ) -> HexagonalReadoutBase:
    """Create and return a readout object based on the readout mode and the header information.
    """
    if readout_mode is HexagonalReadoutMode.RECTANGULAR:
        readout = HexagonalReadoutRectangular(*args, padding=header["padding"])
    elif readout_mode is HexagonalReadoutMode.CIRCULAR:
        readout = HexagonalReadoutCircular(*args)
    else:
        raise RuntimeError(f"Unsupported readout mode: {readout_mode}")
    logger.info(f"Readout chip: {readout}")
    return readout


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
    name, args = current_call(num_backward_steps=1)
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
    max_neighbors: int = -1
    pos_recon_algorithm: str = "centroid"
    eta_2pix_rad_sigma: float = 0.127
    eta_2pix_rad_pivot: float = 0.04
    eta_3pix_rad_offset: float = 0.513
    eta_3pix_rad_sigma: float = 0.141
    eta_3pix_rad_pivot: float = 0.05
    eta_3pix_theta_sigma: float = 0.104
    gain_matrix: Optional[CalibrationMatrix] = None
    noise_matrix: Optional[CalibrationMatrix] = None
    pedestal_matrix: Optional[CalibrationMatrix] = None

def reconstruct(
        input_file_path: str,
        suffix: str = ReconstructionDefaults.suffix,
        zero_sup_threshold: int = ReconstructionDefaults.zero_sup_threshold,
        num_neighbors: int = ReconstructionDefaults.num_neighbors,
        max_neighbors: int = ReconstructionDefaults.max_neighbors,
        pos_recon_algorithm: str = ReconstructionDefaults.pos_recon_algorithm,
        eta_2pix_rad_sigma: float = ReconstructionDefaults.eta_2pix_rad_sigma,
        eta_2pix_rad_pivot: float = ReconstructionDefaults.eta_2pix_rad_pivot,
        eta_3pix_rad_offset: float = ReconstructionDefaults.eta_3pix_rad_offset,
        eta_3pix_rad_sigma: float = ReconstructionDefaults.eta_3pix_rad_sigma,
        eta_3pix_rad_pivot: float = ReconstructionDefaults.eta_3pix_rad_pivot,
        eta_3pix_theta_sigma: float = ReconstructionDefaults.eta_3pix_theta_sigma,
        gain_matrix: Optional[CalibrationMatrix] = ReconstructionDefaults.gain_matrix,
        noise_matrix: Optional[CalibrationMatrix] = ReconstructionDefaults.noise_matrix,
        pedestal_matrix: Optional[CalibrationMatrix] = ReconstructionDefaults.pedestal_matrix,
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

    max_neighbors : int
        The maximum number of neighbor pixels to be used for the clustering. If max_neighbors is
        specified (i.e. different from -1), it has priority over num_neighbors.

    pos_recon_algorithm : str
        The position reconstruction algorithm to use.

    eta_2pix_rad_sigma : float
        The sigma parameter for the radial component of the eta function for two pixel events.

    eta_2pix_rad_pivot : float
        The pivot parameter for the radial component of the eta function for two pixel events.

    eta_3pix_rad_offset : float
        The offset parameter for the radial component of the eta function for three pixel events.

    eta_3pix_rad_sigma : float
        The sigma parameter for the radial component of the eta function for three pixel events.

    eta_3pix_rad_pivot : float
        The pivot parameter for the radial component of the eta function for three pixel events.

    eta_3pix_theta_sigma : float
        The sigma parameter for the angular component of the eta function for three pixel events.

    gain_matrix : CalibrationMatrix
        The gain matrix to use for the reconstruction.

    noise_matrix : CalibrationMatrix
        The noise matrix to use for the reconstruction.

    pedestal_matrix : CalibrationMatrix
        The pedestal matrix to use for the reconstruction.
    """
    # Open the input file and extract the header and the readout information.
    input_file, header, readout_mode = open_file(input_file_path)
    # Creating the readout object.
    args = HexagonalLayout(header["layout"]), header["num_cols"], header["num_rows"],\
        header["pitch"], noise_matrix, gain_matrix, pedestal_matrix
    readout = create_readout(readout_mode, header, *args)
    # Define the effective number of neighbors to be used for the clustering. If max_neighbors is
    # specified (i.e. different from -1), it has priority over num_neighbors. It is necessary to
    # define it here because rectangular readout doesn't have a fixed number of neighbors, contrary
    # to the circular.
    effective_neighbors = max_neighbors if max_neighbors >= 0 else num_neighbors
    # Create the dictionary with the reconstruction parameters to be passed to the clustering.
    recon_pars = None
    if pos_recon_algorithm == "eta":
        recon_pars = dict(
            eta_2pix_rad_sigma=eta_2pix_rad_sigma,
            eta_2pix_rad_pivot=eta_2pix_rad_pivot,
            eta_3pix_rad_offset=eta_3pix_rad_offset,
            eta_3pix_rad_sigma=eta_3pix_rad_sigma,
            eta_3pix_rad_pivot=eta_3pix_rad_pivot,
            eta_3pix_theta_sigma=eta_3pix_theta_sigma,
            pitch=header["pitch"]
        )
    # Run the actual reconstruction.
    clustering = ClusteringNN(readout, zero_sup_threshold, effective_neighbors,
                              pos_recon_algorithm, recon_pars)
    output_file_path = input_file_path.replace(".h5", f"_{suffix}.h5")
    output_file = ReconOutputFile(output_file_path)
    if header_kwargs is not None:
        output_file.update_header(**header_kwargs)
    output_file.update_digi_header(**input_file.header)
    # Create a list of acceptable cluster sizes.
    size = list(range(1, max_neighbors + 2)) if max_neighbors >= 0 else [num_neighbors + 1]
    for i, event in tqdm(enumerate(input_file)):
        try:
            cluster = clustering.run(event)
        except IndexError as e:
            logger.warning(f"Error reconstructing event with trigger ID {event.trigger_id}: {e}")
        if cluster.size() in size:
            # Need to pass the recon method and other stuff as argument to ReconEvent
            args = event.trigger_id, event.timestamp(), event.livetime, cluster
            recon_event = ReconEvent(*args)
            try:
                mc_event = input_file.mc_event(i)
            except IndexError:
                mc_event = None
            output_file.add_row(recon_event, mc_event)

    output_file.flush()
    input_file.close()
    output_file.close()
    return output_file_path


@dataclass(frozen=True)
class CalibrationEtaDefaults:

    """Default parameters for the eta function calibration task.

    This is a small helper dataclass to help ensure consistency between the main task
    definition in this Python module and the command-line interface.
    """

    num_bins: int = 50
    zero_sup_threshold: int = 30


def calibrate_eta(
        input_file_path: str,
        num_bins: int,
        zero_sup_threshold: int
        ) -> None:
    """Calibrate the eta function using the events from a digi file.

    Arguments
    ---------
    input_file_path : str
        The path to the input file.

    num_bins : int
        The number of bins to be used in the calibration.

    zero_sup_threshold : int
        The zero-suppression threshold to be used for the clustering in the calibration.
    """
    input_file, header, readout_mode = open_file(input_file_path)
    args = HexagonalLayout(header["layout"]), header["num_cols"], header["num_rows"],\
        header["pitch"], header["enc"], header["gain"], header.get("offset", 0)
    readout = create_readout(readout_mode, header, *args)
    clustering = ClusteringNN(readout, zero_sup_threshold, num_neighbors=6,
                              pos_recon_algorithm="centroid")
    # Create the lists to store the data.
    size_list, photon_pos_list, versors_list, eta_list = [], [], [], []
    # Loop over the events and calculate the interesting quantities.
    for i, event in tqdm(enumerate(input_file)):
        try:
            cluster = clustering.run(event)
        except IndexError:
            continue
        # Analyze only 2-pixel and 3-pixel events.
        if cluster.size() == 2 or cluster.size() == 3:
            mc_event = input_file.mc_event(i)
            size_list.append(cluster.size())
            # Calculate the photon position with respect to the most charged pixel
            ph_pos = np.array([mc_event.absx - cluster.x[0],
                               mc_event.absy - cluster.y[0]]) / header["pitch"]
            photon_pos_list.append(ph_pos)
            eta_list.append(cluster.calculate_eta())
            versors_list.append(cluster.versors())
    input_file.close()
    # Convert the lists to numpy arrays
    size = np.asarray(size_list, dtype=int)
    photon_pos = np.asarray(photon_pos_list, dtype=float)
    versors = np.asarray(versors_list, dtype=float)
    eta = np.asarray(eta_list, dtype=object)

    # 2-pixel events calibration
    mask_2pix = size == 2
    eta_2pix = eta[mask_2pix].flatten()
    dr_2pix = distance(photon_pos[mask_2pix], versors[mask_2pix, 0])
    calibrate_dr_2pix(eta_2pix, dr_2pix, nbins=num_bins)

    # 3-pixel events calibration
    mask_3pix = size == 3
    eta_3pix = np.stack(eta[mask_3pix])
    dr_3pix = distance(photon_pos[mask_3pix])
    theta_3pix = angle(photon_pos[mask_3pix], versors[mask_3pix])
    calibrate_dr_3pix(eta_3pix, dr_3pix, nbins=num_bins)
    calibrate_theta_3pix(eta_3pix, dr_3pix, theta_3pix, nbins=num_bins)
    plt.show()


def calibrate_noise(
        input_file_path: str
        ) -> str:
    """Calibrate noise of the readout chip using the events from a digi file.
    The results are stored as a matrix in a HDF5 file.

    Arguments
    ---------
    input_file_path : str
        The path to the input file.
    """
    # Open the input file and extract the readout information.
    input_file, header, readout_mode = open_file(input_file_path)
    # The analysis is only supported for rectangular readout.
    if readout_mode is not HexagonalReadoutMode.RECTANGULAR:
        raise RuntimeError("Noise calibration is only supported for rectangular readout")
    # Create the calibration matrix
    noise_matrix = CalibrationMatrix(header["num_cols"], header["num_rows"])
    noise_calibration = CalibrateNoise(noise_matrix)
    # Loop over the events and analyze the noise.
    for _, event in tqdm(enumerate(input_file)):
        noise_calibration.analyze_event(event)
    noise_calibration.update()
    # Close the input file and save the noise matrix to a HDF5 file.
    output_file_path = input_file_path.replace(".h5", "_matrix_noise.h5")
    noise_matrix.to_hdf5(output_file_path, "noise", False)
    input_file.close()
    return output_file_path


@dataclass(frozen=True)
class CalibrationDarkDefaults:
    """Default parameters for the dark calibration task.

    This is a small helper dataclass to help ensure consistency between the main task
    definition in this Python module and the command-line interface.
    """

    has_source: bool = True
    batch_size: int = 5000000


def calibrate_dark(
        input_file_path: str,
        has_source: bool = CalibrationDarkDefaults.has_source,
        batch_size: int = CalibrationDarkDefaults.batch_size
        ) -> str:
    # Open the input file and extract the readout information.
    input_file, header, readout_mode = open_file(input_file_path)
    # The analysis is only supported for rectangular readout.
    if readout_mode is not HexagonalReadoutMode.RECTANGULAR:
        raise RuntimeError("Noise calibration is only supported for rectangular readout")
    # Create the calibration matrix
    noise_matrix = CalibrationMatrix(header["num_cols"], header["num_rows"])
    pedestal_matrix = CalibrationMatrix(header["num_cols"], header["num_rows"])
    dark_calibration = CalibrateDark(noise_matrix, pedestal_matrix)
    # Loop over the events and analyze the noise.
    for _, event in tqdm(enumerate(input_file)):
        dark_calibration.analyze_event(event, has_source, batch_size)
    # Update the histogram with the last batch of events and fit the data.
    dark_calibration.update_hist()
    dark_calibration.fit()
    # Close the input file and save the noise matrix to a HDF5 file.
    noise_output_file_path = input_file_path.replace(".h5", "_matrix_noise.h5")
    pedestal_output_file_path = input_file_path.replace(".h5", "_matrix_pedestal.h5")
    noise_matrix.to_hdf5(noise_output_file_path, "noise", False)
    pedestal_matrix.to_hdf5(pedestal_output_file_path, "pedestal", False)
    input_file.close()
    return noise_output_file_path


@dataclass(frozen=True)
class CalibrationGainDefaults:
    """Default parameters for the gain calibration task.

    This is a small helper dataclass to help ensure consistency between the main task
    definition in this Python module and the command-line interface.
    """

    num_events: int = 50000
    enc: int = 20
    zero_sup_threshold: int = 20


def calibrate_gain(
        input_file_path: str,
        energy: float,
        noise_matrix: Optional[CalibrationMatrix] = None,
        pedestal_matrix: Optional[CalibrationMatrix] = None,
        num_events: int = CalibrationGainDefaults.num_events,
        zero_sup_threshold: int = CalibrationGainDefaults.zero_sup_threshold,
        ) -> str:
    """Calibrate gain of the readout chip using the events from a digi file.
    The results are stored as a matrix in a HDF5 file.

    Arguments
    ---------
    input_file_path : str
        The path to the input file.

    energy : float
        The energy of the X-ray photons in eV. This is used to convert the charge collected in
        each pixel to the number of electron, which is necessary for the gain calibration.

    noise_matrix : CalibrationMatrix
        The calibration noise matrix to use for the gain calibration.

    num_events : int
        The number of events to simulate to correct the bias.

    zero_sup_threshold : int
        The zero-suppression threshold to use for the clustering in the gain calibration.
    """
    # Open the input file and extract the readout information.
    input_file, header, readout_mode = open_file(input_file_path)
    # Define the arguments to create the readout object with unit gain, necessary for the
    # calibration.
    gain_map = CalibrationMatrix(header["num_cols"], header["num_rows"])
    gain_map.set_value(1.)
    args = HexagonalLayout(header["layout"]), header["num_cols"], header["num_rows"],\
        header["pitch"], noise_matrix, gain_map, pedestal_matrix
    readout = create_readout(readout_mode, header, *args)
    # Initialize the gain matrix and run the calibration.
    gain_matrix = CalibrationMatrix(header["num_cols"], header["num_rows"])
    gain_calibration = CalibrateGain(gain_matrix, energy)
    clustering = ClusteringNN(readout, zero_sup_threshold=zero_sup_threshold, num_neighbors=6,
                              pos_recon_algorithm="centroid")
    for _, event in tqdm(enumerate(input_file)):
        try:
            cluster = clustering.run(event)
        except IndexError:
            continue
        gain_calibration.analyze_cluster(cluster)
    gain_calibration.fit()
    # Create the readout object for the simulation. We are using rectangular readout just because
    # it's faster to simulate.
    gain_sim = CalibrationMatrix(header["num_cols"], header["num_rows"])
    gain_sim.matrix = gain_matrix.matrix
    gain_sim.fill(gain_sim.mean())
    simulation_readout = HexagonalReadoutRectangular(enc=noise_matrix, gain=gain_sim,
                                                     pedestal=pedestal_matrix)
    plt.imshow(gain_matrix.matrix, origin="lower")
    plt.show()
    output = HEXSAMPLE_DATA / "_tmp_simulation_bias.h5"
    # Simulate events with the best-fit gain matrix to correct the bias.
    simulate(
        source=Source(Line(energy), DiskBeam(radius=0.1)),
        sensor=Sensor(),
        readout=simulation_readout,
        num_events=num_events,
        output_file_path=output)
    tmp_input_file = digi_input_file_class("rectangular")(output)
    tmp_gain_matrix = CalibrationMatrix(header["num_cols"], header["num_rows"])
    tmp_gain_calibration = CalibrateGain(tmp_gain_matrix, energy)
    # Re-run the gain calibration on the simulated events to calculate the correction factor.
    for _, event in tqdm(enumerate(tmp_input_file)):
        try:
            cluster = clustering.run(event)
        except IndexError:
            continue
        tmp_gain_calibration.analyze_cluster(cluster)
    tmp_gain_calibration.fit()
    # Calculate the correction factor from the simulation.
    mask = tmp_gain_matrix.hits > 0
    residuals = (tmp_gain_matrix.matrix[mask] - gain_matrix.matrix[mask]) / gain_matrix.matrix[mask]
    # Apply the correction factor to the gain matrix and save it to a HDF5 file.
    gain_matrix.matrix[mask] = gain_matrix.matrix[mask] / (1 + np.mean(residuals))
    output_file_path = input_file_path.replace(".h5", "_matrix_gain.h5")
    gain_matrix.to_hdf5(output_file_path, "gain", False)
    # Close the input files.
    tmp_input_file.close()
    input_file.close()
    return output_file_path


def display(
        input_file_path: str,
        gain_matrix: Optional[CalibrationMatrix] = None,
        noise_matrix: Optional[CalibrationMatrix] = None,
        pedestal_matrix: Optional[CalibrationMatrix] = None,
        ) -> None:
    """Display events from a digi file.

    Arguments
    ---------
    file_path : str
        The path to the digi file.
    zero_sup_threshold : int
        The zero-suppression threshold to use when displaying the digi event.
    event_id : int
        The ID of the event to display. If None, display all events.
    """
    # Open the input file and extract the header and the readout information.
    input_file, header, readout_mode = open_file(input_file_path)
    args = HexagonalLayout(header["layout"]), header["num_cols"], header["num_rows"],\
        header["pitch"], noise_matrix, gain_matrix, pedestal_matrix
    readout = create_readout(readout_mode, header, *args)
    recon_defaults = ReconstructionDefaults
    recon_pars = dict(
        eta_2pix_rad_sigma=recon_defaults.eta_2pix_rad_sigma,
        eta_2pix_rad_pivot=recon_defaults.eta_2pix_rad_pivot,
        eta_3pix_rad_offset=recon_defaults.eta_3pix_rad_offset,
        eta_3pix_rad_sigma=recon_defaults.eta_3pix_rad_sigma,
        eta_3pix_rad_pivot=recon_defaults.eta_3pix_rad_pivot,
        eta_3pix_theta_sigma=recon_defaults.eta_3pix_theta_sigma,
        pitch=header["pitch"]
    )
    _ = EventDisplay(input_file, readout, recon_pars=recon_pars)
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
    # Open the input file
    input_file, _, _ = open_file(input_file_path)
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
    binning = np.linspace(-5. * 0.2, 5. * 0.2, 100)
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
