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

"""Basic simulation, reconstruction and analysis tasks."""

import inspect
import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np
from aptapy.hist import Histogram1d, Histogram2d
from aptapy.models import Line
from aptapy.plotting import plt, setup_gca
from tqdm import tqdm

from . import rng
from .analysis import create_histogram
from .calibration import (
    CALIBRATION_UNITS,
    CalibrateDark,
    CalibrateENC,
    CalibrateEqualization,
    CalibrateGain,
    CalibrateMLE,
    CalibrateNoise,
    CalibrationMatrix,
    CalibrationMetadata,
    CalibrationType,
    MLECalibrationData,
    MLECalibrationMetadata,
)
from .clustering import ClusteringHex, ClusteringNN
from .display import EventDisplay
from .eta import (
    angle,
    calibrate_dr_2pix,
    calibrate_dr_3pix,
    calibrate_theta_3pix,
    distance,
)
from .fileio import (
    DigiInputFileBase,
    ReconInputFile,
    ReconOutputFile,
    digi_input_file_class,
    digi_output_file_class,
    peek_readout_type,
)
from .hexagon import HexagonalGrid, HexagonalLayout
from .logging_ import logger
from .mc import PhotonList
from .pdf import SpectrumPDF
from .readout import (
    AbstractReadout,
    HexagonalReadoutBase,
    HexagonalReadoutCircular,
    HexagonalReadoutMode,
    HexagonalReadoutRectangular,
)
from .recon import ReconEvent
from .sensor import Sensor
from .source import Source
from .xpol import chip_descriptor

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
    """Open a digi file and extract the header and the readout type."""
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


def create_readout(readout_mode: HexagonalReadoutMode, header: dict, *args) -> HexagonalReadoutBase:
    """Create and return a readout object based on the readout mode and the header information."""
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
    zero_sup_threshold: float = 0.0
    num_neighbors: int = 2
    max_neighbors: int = -1
    pos_recon_algorithm: str = "centroid"
    eta_2pix_rad_sigma: float = 0.127
    eta_2pix_rad_pivot: float = 0.04
    eta_3pix_rad_offset: float = 0.513
    eta_3pix_rad_sigma: float = 0.141
    eta_3pix_rad_pivot: float = 0.05
    eta_3pix_theta_sigma: float = 0.104
    mle_data: Optional[MLECalibrationData] = None


def reconstruct(
        input_file_path: str,
        noise_matrix: CalibrationMatrix,
        pedestal_matrix: CalibrationMatrix,
        equalization_matrix: CalibrationMatrix,
        suffix: str = ReconstructionDefaults.suffix,
        zero_sup_threshold: float = ReconstructionDefaults.zero_sup_threshold,
        num_neighbors: int = ReconstructionDefaults.num_neighbors,
        max_neighbors: int = ReconstructionDefaults.max_neighbors,
        pos_recon_algorithm: str = ReconstructionDefaults.pos_recon_algorithm,
        eta_2pix_rad_sigma: float = ReconstructionDefaults.eta_2pix_rad_sigma,
        eta_2pix_rad_pivot: float = ReconstructionDefaults.eta_2pix_rad_pivot,
        eta_3pix_rad_offset: float = ReconstructionDefaults.eta_3pix_rad_offset,
        eta_3pix_rad_sigma: float = ReconstructionDefaults.eta_3pix_rad_sigma,
        eta_3pix_rad_pivot: float = ReconstructionDefaults.eta_3pix_rad_pivot,
        eta_3pix_theta_sigma: float = ReconstructionDefaults.eta_3pix_theta_sigma,
        mle_data: Optional[MLECalibrationData] = ReconstructionDefaults.mle_data,
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

    noise_matrix : CalibrationMatrix
        The noise matrix to use for the reconstruction.

    pedestal_matrix : CalibrationMatrix
        The pedestal matrix to use for the reconstruction.

    equalization_matrix : CalibrationMatrix
        The equalization matrix to use for the reconstruction.

    suffix : str
        The suffix to append to the output file name.

    zero_sup_threshold : float
        The zero-suppression threshold as a multiple of the noise.

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

    mle_data : MLECalibrationData, optional
        MLE calibration data.
    """
    # Open the input file and extract the header and the readout information.
    input_file, header, readout_mode = open_file(input_file_path)
    # Creating the readout object.
    args = HexagonalLayout(header["layout"]), header["num_cols"], header["num_rows"],\
        header["pitch"], noise_matrix, equalization_matrix, pedestal_matrix
    readout = create_readout(readout_mode, header, *args)
    # Create the output file and update the header with the relevant metadata.
    output_file_path = input_file_path.replace(".h5", f"_{suffix}.h5")
    output_file = ReconOutputFile(output_file_path)
    if header_kwargs is not None:
        output_file.update_header(**header_kwargs)
    output_file.update_digi_header(**input_file.header)
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
            pitch=header["pitch"],
        )
    if pos_recon_algorithm == "mle":
        if mle_data is None:
            raise RuntimeError("MLE data must be provided for MLE position reconstruction")
        recon_pars = dict(
            mle_data=mle_data,
            noise_matrix=noise_matrix,
            equalization_matrix=equalization_matrix,
            pitch=header["pitch"]
        )
        clustering = ClusteringHex(readout, 0, pos_recon_algorithm, recon_pars)
        num_neighbors = 6
    else:
        clustering = ClusteringNN(
            readout, zero_sup_threshold, effective_neighbors, pos_recon_algorithm, recon_pars
        )
    # Run the actual reconstruction.
    # Create a list of acceptable cluster sizes.
    size = list(range(1, max_neighbors + 2)) if max_neighbors >= 0 else [num_neighbors + 1]
    for i, event in tqdm(enumerate(input_file)):
        try:
            cluster = clustering.run(event)
        except IndexError as e:
            logger.warning(f"Error reconstructing event with trigger ID {event.trigger_id}: {e}")
        if cluster is not None and (cluster.size() in size):
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


def calibspec(input_file_path: str) -> str:
    """Create a probability density function from a reconstructed spectrum to use in
    the gain calibration.

    Arguments
    ----------
    input_file_path : str
        The path to the input recon file.
    """
    name, args = current_call(1)
    logger.info(f"Running {__name__}.{name} with arguments {args}...")
    input_file = ReconInputFile(input_file_path)
    energy = input_file.column("energy")
    input_file.close()
    pdf = SpectrumPDF()
    logger.info("Fitting the PDF to the energy distribution...")
    pdf.fit(energy)
    output_file_path = input_file_path.replace(".h5", ".npz")
    logger.info(f"Saving the PDF to {output_file_path}...")
    pdf.to_file(output_file_path)
    return output_file_path


@dataclass(frozen=True)
class CalibrationMLEDefaults:
    """Default parameters for the Maximum Likelihood Estimator (MLE) calibration task.

    This is a small helper dataclass to help ensure consistency between the main task
    definition in this Python module and the command-line interface.
    """

    bin_size: float = 0.01


def calibrate_mle(
        input_file_path: str,
        noise_matrix: CalibrationMatrix,
        pedestal_matrix: CalibrationMatrix,
        equalization_matrix: CalibrationMatrix,
        bin_size: float,
    ) -> str:
    """Calibrate the charge diffusion for the Maximum Likelihood Estimator (MLE) position
    reconstruction algorithm, using the events from a digi file.
    The results are stored as a matrix in a HDF5 file.

    Arguments
    ---------
    input_file_path : str
        The path to the input file.
    bin_size : float
        The size of the bins to use for the charge diffusion matrix.
    """
    # Open the input file and extract the header and the readout information.
    input_file, header, readout_mode = open_file(input_file_path)
    grid_args = HexagonalLayout(header["layout"]), header["num_cols"], header["num_rows"],\
        header["pitch"]
    grid = HexagonalGrid(*grid_args)
    # Create the readout object, necessary to create the clustering object
    readout_args = *grid_args, noise_matrix, equalization_matrix, pedestal_matrix
    readout = create_readout(readout_mode, header, *readout_args)
    # To correctly analyze every type of event, we need a zero suppression threshold
    # of 0, because the calibration should be performed on zero-noise simulations.
    clustering = ClusteringHex(readout, zero_sup_threshold=0)
    # Initialize the MLE calibrator and run the event loop.
    mle_calibrator = CalibrateMLE(bin_size, grid)
    logger.info("Starting the event loop...")
    for i, event in tqdm(enumerate(input_file)):
        cluster = clustering.run(event)
        mc_event = input_file.mc_event(i)
        mle_calibrator.analyze_cluster(cluster, mc_event)
    # Close the input file.
    input_file.close()
    data = mle_calibrator.fit()
    # Access sensor information from the header and update the metadata.
    data.update_metadata(MLECalibrationMetadata.PITCH.value, header["pitch"])
    data.update_metadata(MLECalibrationMetadata.LAYOUT.value, header["layout"].value)
    data.update_metadata(MLECalibrationMetadata.DIFFUSION_SIGMA.value, header["diffusion_sigma"])
    data.update_metadata(MLECalibrationMetadata.THICKNESS.value, header["thickness"])
    # Save the calibration results to a HDF5 file.
    output_file_path = input_file_path.replace(".h5", "_mle_matrices.h5")
    logger.info(f"Saving the MLE calibration data to {output_file_path}...")
    data.to_hdf5(output_file_path, CalibrationType.MLE)
    logger.info("Done!")
    return output_file_path


@dataclass(frozen=True)
class CalibrationEtaDefaults:
    """Default parameters for the eta function calibration task.

    This is a small helper dataclass to help ensure consistency between the main task
    definition in this Python module and the command-line interface.
    """

    num_bins: int = 50
    zero_sup_threshold: float = 1.0


def calibrate_eta(
    input_file_path: str,
    noise_matrix: CalibrationMatrix,
    pedestal_matrix: CalibrationMatrix,
    equalization_matrix: CalibrationMatrix,
    num_bins: int = CalibrationEtaDefaults.num_bins,
    zero_sup_threshold: float = CalibrationEtaDefaults.zero_sup_threshold,
) -> None:
    """Calibrate the eta function using the events from a digi file.

    Arguments
    ---------
    input_file_path : str
        The path to the input file.

    noise_matrix : CalibrationMatrix
        The noise calibration matrix to use for the analysis.

    pedestal_matrix : CalibrationMatrix
        The pedestal calibration matrix to use for the analysis.

    equalization_matrix : CalibrationMatrix
        The equalization calibration matrix to use for the analysis.

    num_bins : int
        The number of bins to be used in the calibration.

    zero_sup_threshold : float
        The zero-suppression threshold as a multiple of the noise.
    """
    input_file, header, readout_mode = open_file(input_file_path)
    args = (
        HexagonalLayout(header["layout"]),
        header["num_cols"],
        header["num_rows"],
        header["pitch"],
        noise_matrix,
        equalization_matrix,
        pedestal_matrix,
    )
    readout = create_readout(readout_mode, header, *args)
    clustering = ClusteringNN(
        readout, zero_sup_threshold, num_neighbors=6, pos_recon_algorithm="centroid"
    )
    # Create the lists to store the data.
    size_list, photon_pos_list, versors_list, eta_list = [], [], [], []
    # Loop over the events and calculate the interesting quantities.
    for i, event in tqdm(enumerate(input_file)):
        try:
            cluster = clustering.run(event)
        except IndexError:
            continue
        # Analyze only 2-pixel and 3-pixel events.
        if cluster is not None and (cluster.size() == 2 or cluster.size() == 3):
            mc_event = input_file.mc_event(i)
            size_list.append(cluster.size())
            # Calculate the photon position with respect to the most charged pixel
            ph_pos = (
                np.array([mc_event.absx - cluster.x[0], mc_event.absy - cluster.y[0]])
                / header["pitch"]
            )
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


@dataclass(frozen=True)
class SynthesizeCalibrationDefaults:
    """Default values for the generate_calibration_file task."""

    percent_rms: int = 0
    output_dir: Union[str, pathlib.Path] = HEXSAMPLE_DATA
    chip_name: str = "xpol3"
    version: int = 1
    random_seed: int = None


def synthesize_calibration_file(
    calibration_type: CalibrationType,
    mean: float,
    percent_rms: int = SynthesizeCalibrationDefaults.percent_rms,
    chip_name: str = SynthesizeCalibrationDefaults.chip_name,
    output_dir: Union[str, pathlib.Path] = SynthesizeCalibrationDefaults.output_dir,
    version: int = SynthesizeCalibrationDefaults.version,
    random_seed: int = SynthesizeCalibrationDefaults.random_seed,
) -> str:
    """Generate a synthetic calibration file for the given calibration type and
    chip name.

    Arguments
    ---------
    calibration_type : CalibrationType
        The type of calibration to generate.

    mean : float
        The mean value of the sample distribution.

    percent_rms : int, optional
        The root mean square of the sample distribution, expressed as a percentage
        of the mean. note we treat this as an integer, assuming that we shall
        never be in the situation where we need a very precise fine tuning.

    chip_name : str, optional
        The name of the chip for which to generate the calibration file.

    output_dir : str, optional
        The directory where to save the generated calibration file.

    version : int, optional
        The version number the generated calibration file.

    random_seed : int, optional
        The seed for the random number generator.
    """
    name, args = current_call(1)
    logger.info(f"Running {__name__}.{name} with arguments {args}...")
    # Initialize the random number generator with the given seed
    rng.initialize(seed=random_seed)
    num_cols, num_rows = chip_descriptor(chip_name).size
    # Generate the file name
    file_name = f"sim_{chip_name}_{calibration_type.value}-{mean:g}".replace(".", "p")
    # Append the RMS information to the file name
    if percent_rms > 0:
        file_name += f"_gauss-p{percent_rms:02d}".replace(".", "p")
    elif percent_rms == 0:
        file_name += "_uniform"
    else:
        raise ValueError("Percent RMS must be non-negative")
    # Append the version number to the file name
    file_name += f"_v{version:03d}.h5"
    # Generate the calibration matrix with the appropriate size and values
    calibration_matrix = CalibrationMatrix(num_cols, num_rows)
    rms = mean * percent_rms / 100
    if calibration_type == CalibrationType.EQUALIZATION:
        calibration_matrix.update_metadata(CalibrationMetadata.ADC_TO_EV, mean)
        rms /= mean
        mean = 1.0
    logger.info(
        f"Generating {calibration_type.value} calibration matrix with "
        f"mean {mean:g} and RMS {rms:g}..."
    )
    calibration_matrix.values = rng.generator.normal(mean, scale=rms, size=(num_rows, num_cols))
    # Save the calibration matrix to the output directory
    output_path = pathlib.Path(output_dir) / file_name
    logger.info(f"Saving to {output_path}...")
    calibration_matrix.to_hdf5(output_path, calibration_type, True)
    logger.info("Done!")
    return str(output_path)


def calibrate_noise(input_file_path: str) -> str:
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
    # Create the object to calibrate the noise and run the analysis.
    noise_calibration = CalibrateNoise(header["num_cols"], header["num_rows"])
    # Loop over the events and analyze the noise.
    logger.info("Starting the event loop...")
    for _, event in tqdm(enumerate(input_file)):
        noise_calibration.analyze_event(event)
    logger.info("Calculating the noise matrix...")
    noise_matrix = noise_calibration.fit()
    # Close the input file and save the noise matrix to a HDF5 file.
    output_file_path = input_file_path.replace(".h5", "_matrix_noise.h5")
    logger.info(f"Saving to {output_file_path}...")
    noise_matrix.to_hdf5(output_file_path, CalibrationType.NOISE, False)
    input_file.close()
    logger.info("Done!")
    return output_file_path


@dataclass(frozen=True)
class CalibrationDarkDefaults:
    """Default parameters for the dark calibration task.

    This is a small helper dataclass to help ensure consistency between the main task
    definition in this Python module and the command-line interface.
    """

    algorithm: str = "welford"
    has_source: bool = True
    batch_size: int = 5000000


def calibrate_dark(
    input_file_path: str,
    algorithm: str = CalibrationDarkDefaults.algorithm,
    has_source: bool = CalibrationDarkDefaults.has_source,
    batch_size: int = CalibrationDarkDefaults.batch_size,
) -> Tuple[str, str]:
    # Open the input file and extract the readout information.
    input_file, header, readout_mode = open_file(input_file_path)
    # The analysis is only supported for rectangular readout.
    if readout_mode is not HexagonalReadoutMode.RECTANGULAR:
        raise RuntimeError("Noise calibration is only supported for rectangular readout")
    # Create the calibration matrix
    dark_calibration = CalibrateDark(header["num_cols"], header["num_rows"], algorithm)
    # Loop over the events and analyze the noise.
    logger.info("Starting the event loop...")
    for _, event in tqdm(enumerate(input_file)):
        dark_calibration.analyze_event(event, has_source, batch_size)
    # Update the histogram with the last batch of events and fit the data.
    logger.info("Calculating the noise and pedestal matrices...")
    noise_matrix, pedestal_matrix = dark_calibration.fit()
    # Close the input file and save the noise matrix to a HDF5 file.
    noise_output_file_path = input_file_path.replace(".h5", "_matrix_noise.h5")
    pedestal_output_file_path = input_file_path.replace(".h5", "_matrix_pedestal.h5")
    logger.info(f"Saving noise matrix to {noise_output_file_path}...")
    logger.info(f"Saving pedestal matrix to {pedestal_output_file_path}...")
    noise_matrix.to_hdf5(noise_output_file_path, CalibrationType.NOISE, False)
    pedestal_matrix.to_hdf5(pedestal_output_file_path, CalibrationType.PEDESTAL, False)
    input_file.close()
    logger.info("Done!")
    return noise_output_file_path, pedestal_output_file_path


@dataclass(frozen=True)
class CalibrationEncDefaults:
    """Default parameters for the ENC calibration task.

    This is a small helper dataclass to help ensure consistency between the main task
    definition in this Python module and the command-line interface.
    """

    output_dir: Union[str, pathlib.Path] = HEXSAMPLE_DATA


def calibrate_enc(
    noise_matrix: CalibrationMatrix,
    gain_matrix: CalibrationMatrix,
    output_dir: Union[str, pathlib.Path] = CalibrationEncDefaults.output_dir,
) -> str:
    """Calibrate the equivalent noise charge (ENC) of the readout chip using the noise and gain
    matrices. The results are stored as a matrix in a HDF5 file.

    Arguments
    ---------
    noise_matrix : CalibrationMatrix
        The noise calibration matrix to use for the ENC calibration.

    gain_matrix : CalibrationMatrix
        The gain calibration matrix to use for the ENC calibration.
    """
    name, args = current_call(num_backward_steps=1)
    logger.info(f"Running {__name__}.{name} with arguments {args}...")
    enc_calibration = CalibrateENC(noise_matrix, gain_matrix)
    logger.info("Calculating the ENC matrix...")
    enc_matrix = enc_calibration.fit()
    noise_file_name = noise_matrix.metadata["file_name"]
    enc_file_name = noise_file_name.replace("noise", "enc") + ".h5"
    output_file_path = pathlib.Path(output_dir) / enc_file_name
    logger.info(f"Saving to {output_file_path}...")
    enc_matrix.to_hdf5(output_file_path, CalibrationType.ENC, False)
    logger.info("Done!")
    return output_file_path


@dataclass(frozen=True)
class CalibrationEqualizationDefaults:
    """Default parameters for the pixel equalization calibration task.

    This is a small helper dataclass to help ensure consistency between the main task
    definition in this Python module and the command-line interface.
    """

    algorithm: str = "relative"
    pdf: Optional[SpectrumPDF] = None
    size: int = 10
    zero_sup_threshold: float = 1.0


def calibrate_equalization(
    input_file_path: str,
    noise_matrix: CalibrationMatrix,
    pedestal_matrix: CalibrationMatrix,
    algorithm: str = CalibrationEqualizationDefaults.algorithm,
    pdf: Optional[SpectrumPDF] = CalibrationEqualizationDefaults.pdf,
    size: int = CalibrationEqualizationDefaults.size,
    zero_sup_threshold: float = CalibrationEqualizationDefaults.zero_sup_threshold,
) -> str:
    """Calibrate pixel equalization of the readout chip using the events from a digi file.
    The results are stored as a matrix in a HDF5 file.

    Arguments
    ---------
    input_file_path : str
        The path to the input file.

    noise_matrix : CalibrationMatrix
        The calibration noise matrix to use for the pixel equalization calibration.

    pedestal_matrix : CalibrationMatrix
        The pedestal matrix to use for the pixel equalization calibration.

    algorithm : str, optional
        The algorithm to use for the pixel equalization calibration. Supported values
        are "relative" and "absolute".

    pdf : SpectrumPDF, optional
        The spectrum probability density function to use for the pixel equalization
        calibration.

    size : int, optional
        The length of the square region of the chip to fit simultaneously during the
        pixel equalization calibration.

    zero_sup_threshold : float, optional
        The zero-suppression threshold to use for the clustering in the pixel
        equalization calibration.
    """
    # Open the input file and extract the readout information.
    input_file, header, readout_mode = open_file(input_file_path)
    num_cols, num_rows = header["num_cols"], header["num_rows"]
    # Define the arguments to create the readout object with uniform pixel equalization,
    # necessary for the calibration.
    unit_gain_map = CalibrationMatrix(num_cols, num_rows)
    unit_gain_map.set_value(1.0)
    unit_gain_map.update_metadata(CalibrationMetadata.ADC_TO_EV, 1.0)
    args = (
        HexagonalLayout(header["layout"]),
        num_cols,
        num_rows,
        header["pitch"],
        noise_matrix,
        unit_gain_map,
        pedestal_matrix,
    )
    readout = create_readout(readout_mode, header, *args)
    # Initialize the equalization matrix and run the calibration.
    equalization_calibration = CalibrateEqualization(header["num_cols"], header["num_rows"], pdf)
    clustering = ClusteringNN(
        readout,
        zero_sup_threshold=zero_sup_threshold,
        num_neighbors=6,
        pos_recon_algorithm="centroid",
    )
    equalization_calibration = CalibrateEqualization(num_cols, num_rows, algorithm, pdf)
    logger.info("Starting the event loop...")
    for _, event in tqdm(enumerate(input_file)):
        try:
            cluster = clustering.run(event)
        except IndexError:
            continue
        if cluster is not None:
            equalization_calibration.analyze_cluster(cluster)
    logger.info("Calculating the equalization matrix...")
    equalization_matrix = equalization_calibration.fit(size=size)
    output_file_path = input_file_path.replace(".h5", "_matrix_equalization.h5")
    logger.info(f"Saving equalization matrix to {output_file_path}...")
    equalization_matrix.to_hdf5(output_file_path, CalibrationType.EQUALIZATION, False)
    input_file.close()
    logger.info("Done!")
    return output_file_path


@dataclass(frozen=True)
class CalibrationGainDefaults:
    """Default parameters for the gain calibration task.

    This is a small helper dataclass to help ensure consistency between the main task
    definition in this Python module and the command-line interface.
    """

    output_dir: Union[str, pathlib.Path] = HEXSAMPLE_DATA
    material_symbol: str = "Si"


def calibrate_gain(
    equalization_matrix: CalibrationMatrix,
    material_symbol: str = CalibrationGainDefaults.material_symbol,
    output_dir: Union[str, pathlib.Path] = CalibrationGainDefaults.output_dir,
) -> str:
    """Calibrate the gain of the readout chip using the equalization matrix and the
    ionization potential of the sensor material. The results are stored as a matrix
    in a HDF5 file.

    Arguments
    equalization_matrix : CalibrationMatrix
        The equalization matrix to use for the gain calibration.
    material_symbol : str
        The symbol of the sensor material to use for the gain calibration, e.g. "Si" for
        silicon.
    """
    name, args = current_call(num_backward_steps=1)
    logger.info(f"Running {__name__}.{name} with arguments {args}...")
    gain_calibration = CalibrateGain(equalization_matrix, material_symbol)
    logger.info("Calculating the gain matrix...")
    gain_matrix = gain_calibration.fit()
    equalization_file_name = equalization_matrix.metadata["file_name"]
    gain_file_name = equalization_file_name.replace("equalization", "gain") + ".h5"
    output_file_path = pathlib.Path(output_dir) / gain_file_name
    logger.info(f"Saving to {output_file_path}...")
    gain_matrix.to_hdf5(output_file_path, CalibrationType.GAIN, False)
    logger.info("Done!")
    return output_file_path


@dataclass(frozen=True)
class DisplayDefaults:
    """Default parameters for the display task.

    This is a small helper dataclass to help ensure consistency between the main task
    definition in this Python module and the command-line interface.
    """

    noise_matrix: Optional[CalibrationMatrix] = None
    pedestal_matrix: Optional[CalibrationMatrix] = None
    equalization_matrix: Optional[CalibrationMatrix] = None


def display(
    input_file_path: str,
    noise_matrix: Optional[CalibrationMatrix] = DisplayDefaults.noise_matrix,
    pedestal_matrix: Optional[CalibrationMatrix] = DisplayDefaults.pedestal_matrix,
    equalization_matrix: Optional[CalibrationMatrix] = DisplayDefaults.equalization_matrix,
) -> None:
    """Display events from a digi file.

    Arguments
    ---------
    input_file_path : str
        The path to the digi file.

    noise_matrix : CalibrationMatrix, optional
        The noise calibration matrix to use for the display.

    pedestal_matrix : CalibrationMatrix, optional
        The pedestal calibration matrix to use for the display.

    equalization_matrix : CalibrationMatrix, optional
        The equalization calibration matrix to use for the display.
    """
    # Open the input file and extract the header and the readout information.
    input_file, header, readout_mode = open_file(input_file_path)
    cal_matrices = [noise_matrix, equalization_matrix, pedestal_matrix]
    missing = [matrix for matrix in cal_matrices if matrix is None]
    # Check if any of the calibration matrices is missing.
    if 0 < len(missing) < len(cal_matrices):
        logger.warning(
            f"{len(missing)} calibration matrices are missing to perform event reconstruction."
        )
    # Initialize the correct type of event display based on the input matrices.
    grid_args = (
        HexagonalLayout(header["layout"]),
        header["num_cols"],
        header["num_rows"],
        header["pitch"],
    )
    # If any of the calibration matrices is missing, we only show the grid with pixel values.
    if len(missing) > 0:
        grid = HexagonalGrid(*grid_args)
        recon_pars = None
    # If there are no missing calibration matrices, we can also reconstruct the incident photon
    # position and show it in the event display.
    else:
        readout_args = (*grid_args, *cal_matrices)
        grid = create_readout(readout_mode, header, *readout_args)
        recon_defaults = ReconstructionDefaults
        recon_pars = dict(
            eta_2pix_rad_sigma=recon_defaults.eta_2pix_rad_sigma,
            eta_2pix_rad_pivot=recon_defaults.eta_2pix_rad_pivot,
            eta_3pix_rad_offset=recon_defaults.eta_3pix_rad_offset,
            eta_3pix_rad_sigma=recon_defaults.eta_3pix_rad_sigma,
            eta_3pix_rad_pivot=recon_defaults.eta_3pix_rad_pivot,
            eta_3pix_theta_sigma=recon_defaults.eta_3pix_theta_sigma,
            pitch=header["pitch"],
        )
    # Create the event display and show the events.
    EventDisplay(input_file, grid, recon_pars=recon_pars)
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
    name, args = current_call(1)
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
    binning = np.linspace(-5.0 * 0.2, 5.0 * 0.2, 100)
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
    # Closing the file and showing the figures.
    plt.figure("x-direction resolution")
    binning = np.linspace((x - x_mc).min(), (x - x_mc).max(), 100)
    histx = Histogram1d(binning, xlabel=r"$x - x_{MC}$ [cm]").fill(x - x_mc)
    histx.plot()
    plt.figure("y-direction resolution")
    binning = np.linspace((y - y_mc).min(), (y - y_mc).max(), 100)
    histy = Histogram1d(binning, xlabel=r"$y - y_{MC}$ [cm]").fill(y - y_mc)
    histy.plot()
    input_file.close()
    plt.show()


@dataclass(frozen=True)
class CalibviewDefaults:
    """Default parameters for the calibview task.

    This is a small helper dataclass to help ensure consistency between the main task
    definition in this Python module and the command-line interface.
    """

    mc_matrix: Optional[CalibrationMatrix] = None
    min_hits: int = 0
    rel_error: float = np.inf
    lower_quantile: float = 0.0
    upper_quantile: float = 100.0


def calibview(
    matrix: CalibrationMatrix,
    mc_matrix: Optional[CalibrationMatrix] = CalibviewDefaults.mc_matrix,
    min_hits: int = CalibviewDefaults.min_hits,
    rel_error: float = CalibviewDefaults.rel_error,
    lower_quantile: float = CalibviewDefaults.lower_quantile,
    upper_quantile: float = CalibviewDefaults.upper_quantile,
) -> None:
    """Display a calibration matrix and plot some basic statistics about it. If the
    Monte Carlo truth matrix is provided, the correlation between the two matrices
    is also presented.

    Arguments
    ---------
    matrix : CalibrationMatrix
        The calibration matrix to display.

    mc_matrix : CalibrationMatrix, optional
        The Monte Carlo truth calibration matrix to compare with.

    min_hits : int, optional
        The minimum number of hits in a pixel to be included in the statistics.

    rel_error : float, optional
        The maximum relative error in a pixel to be included in the statistics.

    lower_quantile : float, optional
        The lower quantile of the values in the matrix to be included in the statistics.

    upper_quantile : float, optional
        The upper quantile of the values in the matrix to be included in the statistics.
    """
    # pylint: disable=too-many-statements
    name, args = current_call(1)
    logger.info(f"Running {__name__}.{name} with arguments {args}...")
    logger.info("Matrix metadata:")
    # Log the metadata of the matrix.
    for key, value in matrix.metadata.items():
        if isinstance(key, Enum):
            key = key.value
        logger.info(f"  {key}: {value}")
    unit = CALIBRATION_UNITS.get(matrix.metadata["calibration_type"]).value
    # Calculate the quantiles of the values in the matrix to set the limits for the plots.
    rel_error_mask = abs(matrix.errors / matrix.values) < rel_error
    hits_mask = matrix.entries >= min_hits
    mask = rel_error_mask & hits_mask
    if not np.any(mask):
        raise RuntimeError("No valid pixels found with the given quality cuts.")
    lower_bound, upper_bound = np.nanpercentile(
        matrix.values.flatten()[mask.flatten()], [lower_quantile, upper_quantile]
    )
    logger.info(
        f"Quality cuts: min_hits={min_hits}, rel_error<{rel_error}, "
        f"lower_quantile={lower_quantile}, upper_quantile={upper_quantile}"
    )
    logger.info(f"Number of calibrated pixels after quality cuts: {np.sum(mask)}")
    # Plot the values matrix.
    plt.figure(f"Calibrated matrix: {matrix.metadata['file_name']}")
    plt.imshow(matrix.values, origin="upper", vmin=lower_bound, vmax=upper_bound)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.colorbar(label=unit)
    # Plot the distribution of the calibrated values.
    vals = matrix.values.flatten()[mask.flatten()]
    edges = np.linspace(lower_bound, upper_bound, 100)
    vals_hist = Histogram1d(edges, label="Distribution", xlabel=unit).fill(vals)
    plt.figure("Distribution of calibrated values")
    vals_hist.plot(statistics=True)
    plt.legend()
    # If the Monte Carlo truth matrix is provided, plot the matrix and its distribution.
    if mc_matrix is not None:
        mc_vals = mc_matrix.values.flatten()
        mc_unit = CALIBRATION_UNITS.get(mc_matrix.metadata["calibration_type"]).value
        if mc_unit != unit:
            logger.warning(
                f"Unit of the Monte Carlo matrix ({mc_unit}) is different from"
                f" the unit of the calibrated matrix ({unit})."
            )
        # Plot the Monte Carlo truth matrix.
        plt.figure(f"Monte Carlo truth matrix: {mc_matrix.metadata['file_name']}")
        plt.imshow(mc_matrix.values, origin="upper")
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.colorbar(label=mc_unit)
        # Plot the distribution of the Monte Carlo truth values.
        mc_edges = np.linspace(np.nanmin(mc_vals), np.nanmax(mc_vals), 100)
        # If Monte Carlo distribution is uniform, we need to modify the edges
        if mc_edges[0] == mc_edges[-1]:
            val = mc_edges[0]
            mc_edges = np.linspace(val * 0.9, val * 1.1, 100)
        mc_vals_hist = Histogram1d(mc_edges, label="MC Distribution", xlabel=mc_unit).fill(mc_vals)
        plt.figure("Distribution of Monte Carlo truth values")
        mc_vals_hist.plot(statistics=True)
        plt.legend()
        # Plot the correlation between the calibrated values and the Monte Carlo truth values.
        plt.figure("Correlation between calibrated values and Monte Carlo truth values")
        plt.scatter(vals, mc_vals[mask.flatten()], alpha=0.1, s=10)
        line = Line()
        line.intercept.freeze(0.0)
        line.fit(vals, mc_vals[mask.flatten()])
        label = f"Slope: {line.slope.ufloat()}"
        line.plot(
            label=label,
            color="black",
            linestyle="--",
        )
        plt.legend()
        plt.xlabel(f"Calibrated values [{unit}]")
        plt.ylabel(f"Monte Carlo truth values [{mc_unit}]")
        # Plot the residuals distribution.
        residuals = (vals - mc_vals[mask.flatten()]) / mc_vals[mask.flatten()]
        residual_edges = np.linspace(np.nanmin(residuals), np.nanmax(residuals), 100)
        residual_hist = Histogram1d(
            residual_edges, label="Residuals", xlabel="Relative Residual"
        ).fill(residuals)
        plt.figure("Relative residuals distribution")
        residual_hist.plot(statistics=True)
        plt.legend()
        # Plot the pull distribution.
        pull = (vals - mc_vals[mask.flatten()]) / matrix.errors.flatten()[mask.flatten()]
        pull = pull[~np.isnan(pull) & ~np.isinf(pull)]
        pull_edges = np.linspace(np.nanmin(pull), np.nanmax(pull), 100)
        pull_hist = Histogram1d(pull_edges, label="Pull", xlabel="Pull").fill(pull)
        plt.figure("Pull distribution")
        pull_hist.plot(statistics=True)
        plt.legend()
    plt.show()
