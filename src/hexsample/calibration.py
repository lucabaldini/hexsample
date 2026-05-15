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

"""Calibration facilities.
"""

import gc
import inspect
import pathlib
from enum import Enum
from itertools import product
from typing import Iterator, Optional, Tuple

import h5py
import numpy as np
import xraydb
from aptapy.hist import Histogram3d
from aptapy.models import Gaussian, Probit
from iminuit import Minuit
from joblib import Parallel, delayed
from scipy.sparse import csc_matrix, csr_matrix
from tqdm import tqdm

from .clustering import Cluster
from .digi import DigiEventRectangular
from .hexagon import HexagonalGrid
from .mc import MonteCarloEvent
from .pdf import SpectrumPDF
from .position import versor_2pix, versor_3pix, profile
from .stats import RunningStats


class CalibrationType(str, Enum):

    """Enum class expressing the possible calibration types.
    """

    ENC = "enc"
    EQUALIZATION = "equalization"
    ETA = "eta"
    GAIN = "gain"
    NOISE = "noise"
    PEDESTAL = "pedestal"
    POSITION = "position"

    @classmethod
    def values(cls) -> Tuple[str, ...]:
        """Return a tuple with all the enum values.
        """
        return tuple(item.value for item in cls)


class CalibrationMetadata(str, Enum):

    """Enum to store the metadata keys for the calibration matrix.
    """

    FILE_NAME = "file_name"
    NUM_COLS = "num_cols"
    NUM_ROWS = "num_rows"
    NUM_EVENTS = "num_events"
    ENTRIES_AVG = "entries_avg"
    ENTRIES_MIN = "entries_min"
    ENTRIES_MAX = "entries_max"
    NUM_CALIBRATED_PIXELS = "num_calibrated_pixels"
    VERSION = "version"
    CALIBRATION_TYPE = "calibration_type"
    IS_SYNTHETIC = "is_synthetic"
    # This is to convert ADC counts to energy and it's stored only in the
    # equalization matrix, and is measured in eV/ADC count.
    ADC_TO_EV = "adc_to_ev"


class PositionCalibrationMetadata(str, Enum):

    """Enum to store the metadata keys for the MLE calibration data.
    """

    BIN_SIZE = "bin_size"
    CALIBRATION_TYPE = "calibration_type"
    DIFFUSION_SIGMA = "diffusion_sigma"
    FILE_NAME = "file_name"
    LAYOUT = "layout"
    PITCH = "pitch"
    THICKNESS = "thickness"
    VERSION = "version"


class CalibrationUnits(str, Enum):

    """Enum to store the possible units for the calibration matrix values.
    """

    ENC = "Electrons"
    EQUALIZATION = ""
    GAIN = "ADC counts / electron"
    NOISE = "ADC counts"
    PEDESTAL = "ADC counts"


CALIBRATION_UNITS = {
    CalibrationType.ENC: CalibrationUnits.ENC,
    CalibrationType.EQUALIZATION: CalibrationUnits.EQUALIZATION,
    CalibrationType.GAIN: CalibrationUnits.GAIN,
    CalibrationType.NOISE: CalibrationUnits.NOISE,
    CalibrationType.PEDESTAL: CalibrationUnits.PEDESTAL,
}


class CalibrationBase:

    """Base class for calibration data classes.

    This class defines the basic structure and functionalities for the calibration
    classes, such as the storage of the calibration values and metadata, and the
    methods to save and load the calibration data from HDF5 files.

    .. note:: This class is not meant to be used directly, but to be inherited by
              the specific calibration classes.
    """

    VALUES = "values"
    _VALUES_DTYPE = np.float32

    def __init__(self) -> None:
        """Class constructor.
        """
        self._values = None
        self._metadata = {}

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray, np.dtype]]:
        """Iterate over the calibration datasets.
        """
        # Access the class attributes.
        for attr in dir(self):
            # Get only the uppercase attributes.
            if attr.isupper() and not attr.startswith("_"):
                name = getattr(self, attr)
                # Get the corresponding dataset variable name.
                var = f"_{name}"
                if hasattr(self, var):
                    val = getattr(self, var)
                    dtype_var = f"_{attr}_DTYPE"
                    dtype = getattr(self, dtype_var, np.float64) 
                    # Yield the dataset name, value and dtype.
                    yield name, val, dtype

    @property
    def values(self) -> np.ndarray:
        """Return the calibration values.
        """
        if self._values is None:
            raise NotImplementedError("Calibration values have not been initialized yet.")
        return self._values

    @values.setter
    def values(self, new_values: np.ndarray) -> None:
        """Set the calibration values to a new value.
        """
        if new_values.shape != self.values.shape:
            raise ValueError(f"Input matrix has shape {new_values.shape}, but expected shape is "
                             f"{self.values.shape}.")
        self._values = new_values

    @property
    def metadata(self) -> dict:
        """Return the metadata of the calibration.
        """
        raise NotImplementedError("Metadata property is not implemented yet.")

    def update_metadata(self, key: str, value) -> None:
        """Update the metadata dictionary with a new key-value pair.

        Arguments
        ---------
        key : str
            The key of the metadata entry to update.
        value
            The value of the metadata entry to update.
        """
        self._metadata[key] = value

    def to_hdf5(self, file_path: str, calibration_type: CalibrationType) -> str:
        """Save the calibration matrix to an HDF5 file at the given path.

        Arguments
        ---------
        file_path : str
            The path of the file on the disk.
        calibration_type : CalibrationType
            The type of calibration for which the matrix is being saved.
        """
        # Update the metadata with the relevant information for the calibration data.
        self.update_metadata(CalibrationMetadata.FILE_NAME, pathlib.Path(file_path).stem)
        self.update_metadata(CalibrationMetadata.CALIBRATION_TYPE, calibration_type.value)
        # Define the HDF5 compression parameters.
        compression_pars = dict(compression="gzip", compression_opts=9, shuffle=True)
        with h5py.File(file_path, "w") as h5file:
            for name, dataset, dtype in self:
                # If we want to save an array, use the compression...
                if hasattr(dataset, "shape") and dataset.shape != ():
                    h5file.create_dataset(name, data=dataset, dtype=dtype, **compression_pars)
                # ... otherwise, if it's a scalar value, we can't use compression.
                else:
                    h5file.create_dataset(name, data=dataset, dtype=dtype)
            # Save the metadata in the HDF5 file as attributes.
            for key, value in self.metadata.items():
                h5file.attrs[key] = value
        return file_path

    @classmethod
    def from_hdf5(cls, file_path: str) -> "CalibrationBase":
        """Create an instance of the calibration data class from an HDF5 file at
        the given path.

        Arguments
        ---------
        file_path : str
            The path of the file on the disk.
        """
        # Check if the file path is valid and if it exists.
        if file_path is None:
            raise ValueError("No file path provided for the calibration matrix.")
        if not pathlib.Path(file_path).is_file():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        # Open the HDF5 file.
        with h5py.File(file_path, "r") as h5file:
            # Inspect the __init__ method to get the required arguments for the
            # class constructor.
            init_pars = inspect.signature(cls.__init__).parameters
            init_args = {}
            # Open the attributes of the HDF5 file.
            attrs = dict(h5file.attrs)
            # Loop over the required arguments to find the corresponding values
            # in the attributes or datasets of the HDF5 file.
            for key in init_pars:
                if key == "self":
                    continue
                # Check if the argument is in the attributes.
                if key in attrs:
                    init_args[key] = attrs[key]
                # Otherwise recover it from the datasets.
                else:
                    init_args[key] = h5file[key][()]
            # Instantiate the class with the recovered arguments, and set the
            # dataset values and metadata from the HDF5 file.
            obj = cls(**init_args)
            for name, _, _ in obj:
                setattr(obj, f"_{name}", h5file[name][()])
            for key, value in attrs.items():
                obj._metadata[key] = value
                # If some of the metadata keys correspond to public class attributes,
                # set the attribute values as well.
                try:
                    if hasattr(obj, key):
                        setattr(obj, key, value)
                except AttributeError:
                    pass
        return obj

    def __str__(self) -> str:
        """Return a string representation of the calibration data.
        """
        if CalibrationMetadata.FILE_NAME in self._metadata:
            return self._metadata[CalibrationMetadata.FILE_NAME]
        return f"{self.__class__.__name__}(shape={self.values.shape})"


class CalibrationMatrix(CalibrationBase):

    """Class to store and use calibration matrices for the detector readout.

    A calibration matrix is a 2D array with the same shape as the detector readout chip,
    and each element of the matrix represents a pixel.

    Arguments
    ---------
    num_cols : int
        The number of columns of the detector readout chip.
    num_rows : int
        The number of rows of the detector readout chip.
    """

    ENTRIES = "entries"
    ERRORS = "errors"
    _ENTRIES_DTYPE = np.int32
    _ERRORS_DTYPE = np.float32

    def __init__(self, num_cols: int, num_rows: int) -> None:
        """Class constructor.
        """
        super().__init__()
        self._shape = (num_rows, num_cols)
        # Create the arrays to store the calibration data and the number of events for each pixel.
        self._values = np.full(self._shape, np.nan)
        self._entries = np.zeros(self._shape, dtype=int)
        self._errors = np.full(self._shape, np.nan)
        # Other useful information for the metadata
        self.num_events = 0
        self._cached = False
        self._metadata = {
            CalibrationMetadata.NUM_COLS: num_cols,
            CalibrationMetadata.NUM_ROWS: num_rows
            }

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the calibration matrix.
        """
        return self._shape

    @property
    def entries(self) -> np.ndarray:
        """Return the number of events for each pixel in the calibration matrix.
        """
        return self._entries

    @entries.setter
    def entries(self, new_entries: np.ndarray) -> None:
        """Set the value of the number of events for each pixel in the calibration matrix to a new
        value.
        """
        # Check the consistency of the shape of the new entries matrix.
        if new_entries.shape != self._shape:
            raise ValueError(f"Input entries matrix has shape {new_entries.shape}, but expected "
                             f"shape is {self._shape}.")
        self._entries = new_entries
        # Invalidate the cached metadata since the number of events has changed.
        self._cached = False

    @property
    def errors(self) -> np.ndarray:
        """Return the error of the calibration matrix for each pixel.
        """
        return self._errors

    @errors.setter
    def errors(self, new_error: np.ndarray) -> None:
        """Set the value of the error of the calibration matrix to a new value.
        """
        # Check the consistency of the shape of the new error matrix.
        if new_error.shape != self._shape:
            raise ValueError(f"Input error matrix has shape {new_error.shape}, but expected shape "
                             f"is {self._shape}.")
        self._errors = new_error
        # Invalidate the cached metadata since the error values have changed.
        self._cached = False

    @property
    def metadata(self) -> dict:
        """Return the metadata of the calibration matrix.
        """
        if self._cached:
            return self._metadata
        mask = self._entries > 0
        # If there are no pixels with events, we can set the average, minimum and maximum number of
        # events to zero.
        if np.any(mask):
            entries_avg = int(self._entries[mask].mean())
            entries_min = min(self._entries[mask])
            entries_max = max(self._entries[mask])
        else:
            entries_avg = entries_min = entries_max = 0
        # Setting the metadata values.
        self._metadata[CalibrationMetadata.NUM_EVENTS] = self.num_events
        self._metadata[CalibrationMetadata.ENTRIES_AVG] = entries_avg
        self._metadata[CalibrationMetadata.ENTRIES_MIN] = entries_min
        self._metadata[CalibrationMetadata.ENTRIES_MAX] = entries_max
        self._metadata[CalibrationMetadata.NUM_CALIBRATED_PIXELS] = int(mask.sum())
        self._cached = True
        return self._metadata

    def set_value(self, value: float) -> None:
        """Set a value for all the pixels in the calibration matrix.

        Arguments
        ---------
        value : float
            The value to be set for all the pixels in the calibration matrix.
        """
        self._values = np.full(self._shape, value)
        self._cached = False

    def fill(self, value: float, max_hits: int = 0) -> None:
        """Substitute the value of the pixels with less hits than or equal to a certain threshold
        with a given value.

        This is useful to fill the pixels that may not have enough statistics or have zero events.

        Arguments
        ---------
        value : float
            The value to be set for the pixels with less hits than or equal to the threshold.
        max_hits : int
            The maximum number of hits for a pixel to be considered for replacement.
        """
        if not self._metadata.get(CalibrationMetadata.IS_SYNTHETIC, False):
            self._values = np.where(self._entries <= max_hits, value, self._values)
            self._cached = False

    def mean(self, min_hits: int = 1) -> float:
        """Return the mean value of the calibration matrix, calculated as the mean of the pixels
        with at least one event.

        Arguments
        ---------
        min_hits : int
            The minimum number of hits for a pixel to be considered for the mean calculation.
        """
        if self._metadata.get(CalibrationMetadata.IS_SYNTHETIC, False):
            return self._values.mean()
        if not np.any(self._entries >= min_hits):
            return np.nan
        return self._values[self._entries >= min_hits].mean()

    def median(self) -> float:
        """Return the median value of the calibration matrix, calculated as the median of the
        pixels with at least one event.
        """
        if self._metadata.get(CalibrationMetadata.IS_SYNTHETIC, False):
            return np.median(self._values)
        if not np.any(self._entries > 0):
            return np.nan
        return np.median(self._values[self._entries > 0])

    def std(self) -> float:
        """Return the standard deviation of the calibration matrix, calculated as the standard
        deviation of the pixels with at least one event.
        """
        if self._metadata.get(CalibrationMetadata.IS_SYNTHETIC, False):
            return np.std(self._values)
        if not np.any(self._entries > 0):
            return np.nan
        return np.std(self._values[self._entries > 0])

    def to_hdf5(self, file_path: str, calibration_type: CalibrationType, is_synthetic: bool) -> str:
        """Save the calibration matrix to an HDF5 file at the given path.

        Arguments
        ---------
        file_path : str
            The path of the file on the disk.
        calibration_type : CalibrationType
            The type of calibration for which the matrix is being saved.
        is_synthetic : bool
            Whether the calibration data is synthetic or not.
        """
        # pylint: disable=arguments-differ
        self.update_metadata(CalibrationMetadata.IS_SYNTHETIC, is_synthetic)
        self.update_metadata(CalibrationMetadata.NUM_EVENTS, self.num_events)
        return super().to_hdf5(file_path, calibration_type)

    def __call__(self, col: np.ndarray, row: np.ndarray) -> float:
        """Return the value of the calibration matrix for the given pixel coordinates.

        Arguments
        ---------
        col : np.ndarray
            The column coordinates of the pixels to access.
        row : np.ndarray
            The row coordinates of the pixels to access.
        """
        return self.values[row, col]


class PositionCalibrationData(CalibrationBase):

    """Class to store and use calibration data for Maximum Likelihood Estimation
    (MLE) position reconstruction.

    This class stores a set of seven matrices, each containing the average fraction
    of charge collected by a pixel in a cluster of 7 pixels as a function of the incident
    position of the photon on the central pixel of the cluster.
    """

    # MLE calibration data keys and dtypes.
    X_BINS = "x_bins"
    Y_BINS = "y_bins"
    _X_BINS_DTYPE = np.float32
    _Y_BINS_DTYPE = np.float32
 
    # Eta calibration data keys.
    TWO_PIX_RAD_SIGMA = "two_pix_rad_sigma"
    THREE_PIX_RAD_OFFSET = "three_pix_rad_offset"
    THREE_PIX_RAD_SIGMA = "three_pix_rad_sigma"
    THREE_PIX_THETA_SIGMA = "three_pix_theta_sigma"

    def __init__(self, x_bins: np.ndarray, y_bins: np.ndarray) -> None:
        """Class constructor.
        """
        super().__init__()
        # MLE calibration data.
        self._x_bins = x_bins
        self._y_bins = y_bins
        # Calculate the bin size and the limits of the array.
        self._bin_size = x_bins[1] - x_bins[0]
        self._xlim = (x_bins[0] - self._bin_size / 2, x_bins[-1] + self._bin_size / 2)
        self._ylim = (y_bins[0] - self._bin_size / 2, y_bins[-1] + self._bin_size / 2)
        # Create the tensor to store the calibration data.
        self._values = np.zeros((7, len(x_bins), len(y_bins)))
        # Eta calibration data.
        self._two_pix_rad_sigma = 0.
        self._three_pix_rad_offset = 0.
        self._three_pix_rad_sigma = 0.
        self._three_pix_theta_sigma = 0.
        # Some useful information for the metadata.
        self._metadata = {
            PositionCalibrationMetadata.BIN_SIZE: self._bin_size,
            PositionCalibrationMetadata.CALIBRATION_TYPE: CalibrationType.POSITION.value
        }

    @property
    def x_bins(self) -> np.ndarray:
        """Bin centers in the x axis for the calibration data.
        """
        return self._x_bins

    @property
    def y_bins(self) -> np.ndarray:
        """Bin centers in the y axis for the calibration data.
        """
        return self._y_bins

    @property
    def bin_size(self) -> float:
        """Bin size of the calibration data.
        """
        return self._bin_size

    @property
    def xlims(self) -> Tuple[float, float]:
        """Limits of the x axis for the calibration data.
        """
        return self._xlim

    @property
    def ylims(self) -> Tuple[float, float]:
        """Limits of the y axis for the calibration data.
        """
        return self._ylim

    @property
    def two_pix_rad_sigma(self) -> float:
        """Sigma of the radial distribution of the two-pixel clusters for the eta calibration.
        """
        return self._two_pix_rad_sigma

    @two_pix_rad_sigma.setter
    def two_pix_rad_sigma(self, value: float) -> None:
        """Set the value for the parameter.
        """
        self._two_pix_rad_sigma = value

    @property
    def three_pix_rad_offset(self) -> float:
        """Offset of the radial distribution of the three-pixel clusters for the eta calibration.
        """
        return self._three_pix_rad_offset

    @three_pix_rad_offset.setter
    def three_pix_rad_offset(self, value: float) -> None:
        """Set the value for the parameter.
        """
        self._three_pix_rad_offset = value

    @property
    def three_pix_rad_sigma(self) -> float:
        """Sigma of the radial distribution of the three-pixel clusters for the eta calibration.
        """
        return self._three_pix_rad_sigma

    @three_pix_rad_sigma.setter
    def three_pix_rad_sigma(self, value: float) -> None:
        """Set the value for the parameter.
        """
        self._three_pix_rad_sigma = value

    @property
    def three_pix_theta_sigma(self) -> float:
        """Sigma of the angular distribution of the three-pixel clusters for the eta calibration.
        """
        return self._three_pix_theta_sigma

    @three_pix_theta_sigma.setter
    def three_pix_theta_sigma(self, value: float) -> None:
        """Set the value for the parameter.
        """
        self._three_pix_theta_sigma = value

    @property
    def metadata(self) -> dict:
        """Metadata dictionary containing useful information about the calibration data.
        """
        return self._metadata


class CalibrateBase:

    """Base class for the calibration of the detector readout. This class is not meant to be used
    directly, but to be inherited by the specific calibration classes.
    """

    def __init__(self, num_cols: int, num_rows: int) -> None:
        """Class constructor.
        """
        self.cal_matrix = CalibrationMatrix(num_cols, num_rows)


class CalibrateNoise(CalibrateBase):

    """Calibrate the noise of the detector by analyzing the events in a DigiFile. This class
    takes a CalibrationMatrix object as input, and updates the matrix with the data from
    the file.

    In principle, the dataset used for this task should contain data uniformly distributed
    across the detector.

    Arguments
    ---------
    num_cols : int
        The number of columns of the detector readout chip.
    num_rows : int
        The number of rows of the detector readout chip.
    """

    def __init__(self, num_cols: int, num_rows: int) -> None:
        """Class constructor.
        """
        super().__init__(num_cols, num_rows)
        self._sum2 = np.zeros(self.cal_matrix.shape)

    def _remove_signal(self, event: DigiEventRectangular) -> np.ndarray:
        """Remove the signal pixels from the event pha array.
         
        This is done by setting to zero all the pixels in the ROT and their neighbors.

        Arguments
        ---------
        event : DigiEventRectangular
            The event to be analyzed.
        """
        outer_mask = event.roi.outer_mask(margin=1)
        pha = event.pha.copy()
        pha[~outer_mask] = 0
        return pha

    def _bad_event(self, event: DigiEventRectangular) -> bool:
        """Determine if an event is a bad event, i.e. if it is not suitable for the calibration
        analysis. This is done by applying a cut on the size of the region of interest of the
        event.

        Arguments
        ---------
        event : DigiEventRectangular
            The event to be analyzed.
        """
        # Currently we are selecting only events with a roi size smaller than 200 pixels, which
        # cuts out about 5% of the events. We may choose another criterion in the future.
        roi_shape = event.roi.shape()
        return roi_shape[0] * roi_shape[1] > 200

    def analyze_event(self, event: DigiEventRectangular) -> None:
        """Analyze an event to accumulate the noise values for the calibration matrix.

        Arguments
        ---------
        event : DigiEventRectangular
            The event to be analyzed.
        """
        # If the event is not a bad event, we can use it to update the noise matrix.
        if not self._bad_event(event):
            noise_pha = self._remove_signal(event)
            row_slice, col_slice = event.roi.readout_slice()
            self._sum2[row_slice, col_slice] += noise_pha**2
            self.cal_matrix.entries[row_slice, col_slice][noise_pha > 0] += 1
            self.cal_matrix.num_events += 1

    def fit(self) -> CalibrationMatrix:
        """Update the calibration matrix with the noise values calculated from the data.
        """
        values = self.cal_matrix.values.copy()
        hits = self.cal_matrix.entries
        # If the sum2 array is still zero, it means that no events have been analyzed, so we can
        # set the matrix to the default value for all the pixels.
        if np.array_equal(self._sum2, np.zeros(self.cal_matrix.shape)):
            raise ValueError("No events have been analyzed, cannot update the calibration matrix.")
        with np.errstate(divide='ignore', invalid='ignore'):
            values = np.where(hits > 0, np.sqrt(self._sum2 / hits), values)
            error = np.where(hits > 1, values / np.sqrt(2 * (hits - 1)), self.cal_matrix.errors)
        # Write back through the setter so updates persist on the shared object.
        self.cal_matrix.values = values
        self.cal_matrix.errors = error
        return self.cal_matrix


class CalibrateDark:

    """Calibrate the noise and the pedestal of the detector by analyzing the events in a DigiFile.
    Ideally, this operation should be performed on a dataset without any source signal with a scan
    over the entire readout chip. In case of a dataset with source signal, the signal pixels are
    masked out by setting them to zero, and only the remaining pixels are used for the calibration.

    Arguments
    ---------
    num_cols : int
        The number of columns of the detector readout chip.
    num_rows : int
        The number of rows of the detector readout chip.
    algorithm : str
        The algorithm to be used for the calculation of the noise and pedestal values. The options
        are "welford" and "fit".    
    """

    def __init__(self, num_cols: int, num_rows: int, algorithm: str = "welford") -> None:
        """Class constructor.
        """
        self.noise_cal = CalibrationMatrix(num_cols, num_rows)
        self.pedestal_cal = CalibrationMatrix(num_cols, num_rows)
        self._algorithm = algorithm
        # Check if the noise and pedestal calibration matrices have the same shape.
        num_rows, num_cols = self.noise_cal.shape
        if algorithm == "fit":
            xedges = np.linspace(0, num_cols, num_cols + 1)
            yedges = np.linspace(0, num_rows, num_rows + 1)
            # For now just use a fixed number, but we need to fix this
            zedges = np.linspace(0, 2048, 2049)
            self._histogram = Histogram3d(xedges, yedges, zedges)
            # Batch analysis
            self._pha = []
            self._cols = []
            self._rows = []
        # Welford arrays
        elif algorithm == "welford":
            self._stats = RunningStats(shape=self.noise_cal.shape)
        else:
            raise ValueError(f"Invalid algorithm {algorithm} for the dark calibration. "
                             f"Valid options are 'fit' and 'welford'.")

    def _remove_signal(self, event: DigiEventRectangular) -> np.ndarray:
        """Remove the signal pixels from the event pha array.
         
        This is done by setting to zero all the pixels in the ROT and their neighbors.

        Arguments
        ---------
        event : DigiEventRectangular
            The event to be analyzed.
        """
        outer_mask = event.roi.outer_mask(margin=1)
        pha = event.pha.copy()
        pha[~outer_mask] = 0
        return pha

    def _bad_event(self, event: DigiEventRectangular, max_size: int = 200) -> bool:
        """Determine if an event is a bad event, i.e. if it is not suitable for the calibration
        analysis. This is done by applying a cut on the size of the region of interest of the
        event.

        This is done because in real data we occasionally have large events with
        lots of pixels well above the pedestals, and we don't want to use them
        for the analysis. The default threshold of 200 pixels is chosen because
        cuts out about 5% of the events.

        Arguments
        ---------
        event : DigiEventRectangular
            The event to be analyzed.
        max_size : int
            The maximum size of the region of interest for a valid event.
        """
        return event.roi.size > max_size

    def _fill_hist(self) -> None:
        """Fill the histogram with the accumulated data and update the hits for the pixels that
        have been filled in the histogram.
        """
        if len(self._pha) > 0:
            pha = np.array(self._pha)
            cols = np.array(self._cols)
            rows = np.array(self._rows)
            # Fill the histogram with the accumulated data.
            self._histogram.fill(cols, rows, pha)
            # Update the hits for the pixels that have been filled in the histogram.
            # This operation cannot be done with ar[rows, cols] += 1 because it only updates
            # the value once for repeated indexes.
            np.add.at(self.noise_cal.entries, (rows, cols), 1)
            np.add.at(self.pedestal_cal.entries, (rows, cols), 1)
            # Reset the batch arrays
            self._pha = []
            self._cols = []
            self._rows = []

    def _update_hist(self, pha: np.ndarray, col: np.ndarray, row: np.ndarray,
                     batch_size: int) -> None:
        """Accumulate values to fill the histogram in batch and update it when the batch
        size is large enough.
        """
        # Accumulate the data to fill the histogram in batch.
        self._pha.extend(pha)
        self._cols.extend(col)
        self._rows.extend(row)
        # If the size of the accumulated data is large enough, fill the histogram.
        if len(self._pha) >= batch_size:
            self._fill_hist()

    def analyze_event(self, event: DigiEventRectangular, has_source: bool,
                      batch_size: int = 5000000) -> None:
        """Analyze an event to accumulate the ADC counts of noise pixels to calibrate the noise
        and the pedestal.

        Arguments
        ---------
        event : DigiEventRectangular
            The event to be analyzed.
        has_source : bool
            Whether the event has a source signal.
        batch_size : int
            The size of the batch to be analyzed.
        """
        # If the event is not good, skip the analysis for this event.
        if self._bad_event(event):
            return
        if self._algorithm == "welford":
            # Update the Welford statistics for the pixels in the event.
            outer_mask = event.roi.outer_mask(margin=1)
            offset = (event.roi.min_row, event.roi.min_col)
            self._stats.update(event.pha, offset=offset, mask=outer_mask)
        elif self._algorithm == "fit":
            # Flattened the PHA array with the signal pixels removed.
            pha = self._remove_signal(event) if has_source else event.pha
            # Find the coordinates of the pixels with pha > 0 in the event.
            local_rows, local_cols = np.nonzero(pha > 0)
            pha_values = pha[local_rows, local_cols]
            # Traslate the local coordinates to global coordinates.
            row_slice, col_slice = event.roi.readout_slice()
            global_rows = local_rows + row_slice.start
            global_cols = local_cols + col_slice.start
            self._update_hist(pha_values, global_cols, global_rows, batch_size)
        # Update the number of events for the calibration matrices.
        self.noise_cal.num_events += 1
        self.pedestal_cal.num_events += 1

    def _fit_histogram(self) -> Tuple[CalibrationMatrix, CalibrationMatrix]:
        """Calculate the noise and pedestal values for each pixel by fitting the counts
        distribution with a Gaussian model.
        """
        # Fill the histogram with the last batch of accumulated data, if there is any.
        self._fill_hist()
        # Create copies of noise and pedestal matrices to be updated with the fitted values.
        noise = self.noise_cal.values.copy()
        pedestal = self.pedestal_cal.values.copy()
        noise_err = self.noise_cal.errors.copy()
        pedestal_err = self.pedestal_cal.errors.copy()
        # Fit a Gaussian model to the pixel value distribution for each pixel with enough entries.
        model = Gaussian()
        bin_centers = self._histogram.bin_centers(axis=2)
        cols = range(self.noise_cal.shape[1])
        rows = range(self.noise_cal.shape[0])
        # Calculate the mean and the standard deviation of the pixel value distribution for each
        # pixel to use them as initial values for the fit. This is necessary to speed up the fit.
        mean, std = self._histogram.project_statistics(axis=2)
        for col, row in tqdm(product(cols, rows), total=len(cols)*len(rows), miniters=50):
            counts = self._histogram.content[col, row, :]
            # Check if there are enough entries to perform the fit.
            if counts.sum() > 10:
                # sigma = self._histogram.errors[col, row, :]
                sigma = None
                mu = mean.content[col, row]
                s = std.content[col, row]
                xmin = max(0, mu - 3 * s)
                xmax = mu + 3 * s
                p0 = (1., mu, s)
                try:
                    model.fit(bin_centers, counts, xmin=xmin, xmax=xmax, sigma=sigma, p0=p0)
                except (RuntimeError, np.linalg.LinAlgError):
                    continue
                # Update the noise and pedestal matrices with the fitted values for the pixel.
                noise[row, col] = model.sigma.value
                pedestal[row, col] = model.mu.value
                noise_err[row, col] = model.sigma.error
                pedestal_err[row, col] = model.mu.error
        # Write back the calibration matrices.
        self.noise_cal.values = noise
        self.pedestal_cal.values = pedestal
        self.noise_cal.errors = noise_err
        self.pedestal_cal.errors = pedestal_err
        return self.noise_cal, self.pedestal_cal

    def _fit_welford(self) -> Tuple[CalibrationMatrix, CalibrationMatrix]:
        """Update the noise and pedestal calibration matrices with the values calculated from the
        Welford's algorithm for the pixels that have at least one hit, calulate the errors and
        update the entries.
        """
        counts = self._stats.counts()
        # Calculate the noise and pedestal values.
        self.noise_cal.values = self._stats.std()
        self.pedestal_cal.values = self._stats.mean()
        # Update the entries.
        self.noise_cal.entries = counts
        self.pedestal_cal.entries = counts
        # Calculate the errors for the noise and pedestal values.
        mask = counts > 1
        denominator = np.sqrt(counts - 1, where=mask,
                              out=np.full_like(counts, np.nan, dtype=float))
        np.divide(self.noise_cal.values, np.sqrt(2) * denominator,
                out=self.noise_cal.errors, where=mask)
        np.divide(self.noise_cal.values, denominator,
                out=self.pedestal_cal.errors, where=mask)
        return self.noise_cal, self.pedestal_cal

    def fit(self) -> Tuple[CalibrationMatrix, CalibrationMatrix]:
        """Calculate the noise and pedestal calibration matrices.

        Returns
        -------
        noise_cal : CalibrationMatrix
            Updated calibration matrices for the noise.
        pedestal_cal : CalibrationMatrix
            Updated calibration matrices for the pedestal.
        """
        if self._algorithm == "welford":
            return self._fit_welford()
        if self._algorithm == "fit":
            return self._fit_histogram()
        raise ValueError(f"Invalid algorithm {self._algorithm} for the dark calibration.")


# The following methods are used for the equalization matrix calibration. They cannot be moved
# inside the class because they are used as functions to be executed in parallel, and they
# need to be defined at the top level of the module to be picklable by joblib.


def _likelihood_fit(data: csr_matrix, conv_factor: float, pdf: SpectrumPDF,
                    pdf_derivative: callable) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Perform the likelihood fit for a region of the detector.

    Returns
    -------
    equalization : np.ndarray
        The equalization values for the pixels in the region, calculated from the fit.
    error : np.ndarray
        The error on the equalization values for the pixels in the region, calculated
        from the fit.
    """
    # Define the negative log-likelihood function for the fit.
    def nll(pars):
        total_adc = data @ pars
        p = pdf(total_adc * conv_factor)
        # To avoid negative or zero values in the log, clip p to be larger than 1e-10.
        p_clipped = np.clip(p, 1e-10, None)
        return -np.sum(np.log(p_clipped))
    # Define the gradient of the log-likelihood function for the fit.
    def nll_grad(pars):
        total_adc = data @ pars
        p = pdf(total_adc * conv_factor)
        p_clipped = np.clip(p, 1e-10, None)
        dp = pdf_derivative(total_adc * conv_factor)
        grad = -data.T @ (dp / p_clipped) * conv_factor
        return np.asarray(grad).flatten()
    # Define the initial parameters for the fit.
    init_pars = np.ones(data.shape[1])
    # Initialize the Minuit minimizer.
    m = Minuit(nll, init_pars, grad=nll_grad)
    m.limits = [(1e-10, None) for _ in range(len(init_pars))]
    m.errordef = Minuit.LIKELIHOOD
    m.migrad()
    # If the first fit is not successful, try to perform a second fit. We are not passing
    # the gradient, because the fit should be more robust (but slower).
    if not m.valid:
        m = Minuit(nll, init_pars)
        m.limits = [(1e-10, None) for _ in range(len(init_pars))]
        m.errordef = Minuit.LIKELIHOOD
        m.migrad()
    # If the fit is successful, return the pixel equalization values and their errors,
    # otherwise return None.
    if m.valid:
        equalization = 1 / np.array(m.values)
        error = m.errors / np.array(m.values)**2
        return equalization, error
    return None


def _cut_data(data: csc_matrix, ncols: int, cols: np.ndarray, rows: np.ndarray,
                event_sum: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
    """Cut the data to select only the events that have all the charge contained
    in the pixels of the subregion defined by the input column and row coordinates.
    This is done to ensure that the fit is performed only on events that are fully
    contained in the subregion.

    Arguments
    ---------
    data : csr_matrix
        The sparse matrix containing the pha values for the events in the entire detector.
    cols : np.ndarray
        The column coordinates of the pixels in the subregion.
    rows : np.ndarray
        The row coordinates of the pixels in the subregion.
    event_sum : np.ndarray
        The total ADC count for each event.

    Returns
    -------
    data_active_pixels : csr_matrix
        The sparse matrix containing the pha values for the events in the subregion, with
        only the active pixels.
    mask : np.ndarray
        The boolean mask indicating which pixels in the subregion are active.
    """
    # Calculate the index of the pixels in the flattened array to select the
    # correct columns of the sparse matrix.
    pixels_idxs = (rows * ncols + cols).astype(int)
    # Select all the events but only the columns corresponding to the pixels
    # in the subregion defined by the indices.
    data_region = data[:, pixels_idxs]
    # Re-convert to CSR format for efficient row slicing.
    data_region = data_region.tocsr()
    # Select only the events in the subregion where all the charge is contained
    # in the pixels of the subregion. Add also another cut to select only the
    # events with total charge above 0 ADC, just to be sure to remove empty events.
    sub_region_event_sum = data_region.sum(axis=1).A1
    mask = (sub_region_event_sum == event_sum) & (sub_region_event_sum > 0)
    data_good_events = data_region[mask]
    # We want to remove the pixels in the subregion that never got hit, since
    # only pixels with signal can be fitted. This operation ensures that the
    # number of fit parameters is equal to the number of active pixels.
    pixel_sum = data_good_events.sum(axis=0).A1
    active_mask = pixel_sum > 0
    data_active_pixels = data_good_events[:, active_mask]
    return data_active_pixels, active_mask


def _fit_block(
        r_start: int, c_start: int, size: int, nrows: int, ncols: int, data: csc_matrix,
        event_sum: np.ndarray, conv_factor: float, pdf: SpectrumPDF, pdf_derivative: callable
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Method to perform the fit for a block of pixels in the detector.
    This method is meant to be used as a function to be executed in parallel for different
    blocks of pixels.

    .. note:: This function must be defined outside the class to be picklable and usable
            in parallel execution.

    Returns
    -------
    gain : np.ndarray
        The gain values for the pixels in the block, calculated from the fit.
    error : np.ndarray
        The error on the gain values for the pixels in the block, calculated from the fit.
    rows : np.ndarray
        The row coordinates of the active pixels in the block.
    cols : np.ndarray
        The column coordinates of the active pixels in the block.
    num_events_per_pixel : np.ndarray
        The number of events for each active pixel in the block.
    """
    # Calculate the end row and column indices, taking into account the possibility
    # of being at the end of the chip.
    r_end = min(r_start + size, nrows)
    c_end = min(c_start + size, ncols)
    # Create a meshgrid to get all the coordinates of the pixels in the block.
    cols, rows = np.meshgrid(np.arange(c_start, c_end), np.arange(r_start, r_end))
    cols = cols.flatten()
    rows = rows.flatten()
    # Cut the data to select only the events that have signal in the block.
    data_fit, active_mask = _cut_data(data, ncols, cols, rows, event_sum)
    num_events_per_pixel = np.asarray((data_fit > 0).sum(axis=0)).flatten()
    # Check on the data: if there are no good events or no active pixels,
    # we cannot perform the fit, so we return None.
    if data_fit.nnz == 0 or data_fit.shape[1] == 0:
        return None
    # Fit the data of the block.
    result = _likelihood_fit(data_fit, conv_factor, pdf, pdf_derivative)
    # If the fit is not successful, return None and continue with the next block.
    if result is None:
        return None
    # If the fit is successful, return the equalization values and their errors and the
    # coordinates of the active pixels.
    return result[0], result[1], rows[active_mask], cols[active_mask], num_events_per_pixel


class CalibrateEqualization(CalibrateBase):

    """Calculate the equalization matrix for the detector readout. This matrix is used
    to convert the PHA values to Pulse Invariant (PI). By the definition, the mean value
    of the matrix is 1.0.

    The calibration is peformed by analyzing events in a DigiFile.
    
    Arguments
    ---------
    num_cols : int
        The number of columns of the detector readout chip.
    num_rows : int
        The number of rows of the detector readout chip.
    algorithm : str
        The algorithm to be used for the calculation of the equalization values. The options
        are "absolute" and "relative". The "absolute" algorithm performs a likelihood fit to
        calculate the gain of each pixel, while the "relative" algorithm calculates the
        equalization values by comparing the mean PHA of 1-pixel events.
    pdf : SpectrumPDF
        The probability density function of the spectrum of the events in the dataset, which is
        used for the likelihood fit to calculate the equalization values for the pixels.
    """

    def __init__(self, num_cols: int, num_rows: int, algorithm: str = "relative",
                 pdf: Optional[SpectrumPDF] = None) -> None:
        """Class constructor.
        """
        super().__init__(num_cols, num_rows)
        self._algorithm = algorithm
        # Initialize the data structures for the absolute calibration algorithm.
        if algorithm == "absolute":
            self._event_count = 0
            self._pha = []
            self._coords = []
            self._event_rows = []
            if pdf is None:
                raise ValueError("A SpectrumPDF object must be provided for the "
                                 "absolute equalization calibration.")
            self._pdf = pdf
            self._pdf_derivative = pdf.derivative
        # Initialize the data structures for the relative calibration algorithm.
        elif algorithm == "relative":
            self._stats = RunningStats(shape=self.cal_matrix.shape)
        else:
            raise ValueError(f"Invalid algorithm {algorithm} for the equalization calibration. "
                             f"Valid options are 'absolute' and 'relative'.")

    def analyze_cluster(self, cluster: Cluster) -> None:
        """Analyze the event cluster to update the calibration matrix.
        """
        if self._algorithm == "absolute":
            # Get the coordinates of the cluster pixels
            cols = cluster.col
            rows = cluster.row
            # Update the arrays for the least squares fit.
            self._pha.extend(cluster.pha)
            ncols = self.cal_matrix.shape[1]
            for col, row in zip(cols, rows):
                # Calculate the index of the pixel in the flattened array
                self._coords.append(row * ncols + col)
                self._event_rows.append(self._event_count)
            # Update the event count
            self._event_count += 1
            self.cal_matrix.num_events += 1
        elif self._algorithm == "relative":
            if cluster.size() == 1:
                pha = cluster.pha.reshape((1, 1))
                self._stats.update(pha, offset=(cluster.row[0], cluster.col[0]))
                self.cal_matrix.num_events += 1

    def _fit_absolute(self, size: int) -> CalibrationMatrix:
        """Fit the collected events to determine the gain of each pixel.

        Arguments
        ---------
        size : int
            The length of the square chip region to be fitted simultaneously. The optimal
            value for this parameter is a trade-off between the number of active pixels in
            the chip and the computational time of the fit, which increases significantly
            with the number of pixels. For small active regions, a size of 6 can be a good
            choice, while for the full chip calibration, 10 is a good value. 

        Returns
        -------
        cal_matrix : CalibrationMatrix
            Updated calibration matrix with the gain values calculated from the data.
        """
        # Create a sparse matrix where the rows correspond to the events and the columns
        # correspond to the pixels. In each row, the non-zero entries correspond to the
        # pha values of the pixels that are hit in the event.
        nrows, ncols = self.cal_matrix.shape
        shape = (self._event_count, nrows * ncols)
        # Convert the lists to numpy array of int16 to save memory.
        data_pha = np.array(self._pha, dtype=np.int16)
        data_coords = np.array(self._coords, dtype=np.int32)
        data_event_rows = np.array(self._event_rows, dtype=np.int32)
        # Empty the lists to free memory.
        self._pha = []
        self._coords = []
        self._event_rows = []
        # Create the sparse matrix with the collected data.
        data_csr = csr_matrix((data_pha, (data_event_rows, data_coords)), shape=shape)
        # Calculate the sum of the ADC counts in each event, which is used for the data
        # cuts in the fit, and to calculate the mean ADC count in an event.
        event_sum = data_csr.sum(axis=1).A1
        # Calculate the conversion factor from ADC to eV.
        adc_to_ev = self._pdf.mean() / event_sum.mean()
        # Calculate the starting column and row indices for the blocks. We are using an
        # overlap of 2 pixels between the blocks to ensure that pixels on the edges are
        # correctly calibrated.
        col_starts = np.arange(0, ncols - 2, size - 2)
        row_starts = np.arange(0, nrows - 2, size - 2)
        block_indices = list(product(row_starts, col_starts))
        # Create the arrays to store the weighted gain values and the sum of the weights
        # for each pixel. We will use these arrays to calculate a weighted average of the
        # results, since some pixels are fitted multiple times due to the overlap.
        weighted_gains = np.zeros((nrows, ncols))
        sum_weights = np.zeros((nrows, ncols))
        entries = np.zeros((nrows, ncols), dtype=int)
        # Convert the CSR matrix to CSC format to optimize column slicing during data cuts.
        data_csc = data_csr.tocsc()
        # Free the memory used by the CSR matrix and temporary arrays.
        del data_csr, data_pha, data_coords, data_event_rows
        gc.collect()
        # Start the parallel processing of the blocks.
        args = (size, nrows, ncols, data_csc, event_sum, adc_to_ev,
                self._pdf, self._pdf_derivative)
        bar_format = "{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(_fit_block)(
                r_start, c_start, *args)
                for r_start, c_start in tqdm(block_indices, total=len(block_indices),
                                             bar_format=bar_format)
            )
        # Now we need to aggregate the results from the blocks.
        for _result in results:
            # If the block is empty or the fit is not successful, move to the next block.
            if _result is None:
                continue
            # Unpack the results from the block fit.
            gain, error, rows, cols, num_events_per_pixel = _result
            weights = 1 / (error**2 + 1e-10)
            # Use the mask of the active pixels to update the weighted gains and the sum
            # of the weights matrices for the pixels in the block that have been fitted.
            np.add.at(weighted_gains, (rows, cols), gain * weights)
            np.add.at(sum_weights, (rows, cols), weights)
            np.maximum.at(entries, (rows, cols), num_events_per_pixel)
        # Calculate the final values and errors as the weighted average of the fit results.
        values = np.divide(weighted_gains, sum_weights,
                           out=np.full_like(weighted_gains, np.nan),
                           where=sum_weights > 0)
        errors = np.full_like(weighted_gains, np.nan)
        np.sqrt(sum_weights, out=errors, where=sum_weights > 0)
        np.divide(1, errors, out=errors, where=sum_weights > 0)
        # Write the results back to the calibration matrix.
        self.cal_matrix.values = values
        self.cal_matrix.errors = errors
        self.cal_matrix.entries = entries
        self.cal_matrix.update_metadata(CalibrationMetadata.ADC_TO_EV, adc_to_ev)
        return self.cal_matrix

    def _fit_relative(self) -> CalibrationMatrix:
        """Analyze the collected data to calculate the equalization values
        for the pixels and update the calibration matrix with the calculated values.

        Returns
        -------
        cal_matrix : CalibrationMatrix
            Updated calibration matrix with the equalization values calculated from the data.
        """
        # Get the mean and the standard deviation of the pixel value distribution
        # for each pixel.
        mu = self._stats.mean()
        std = self._stats.std()
        # Calculate the mean value of the pixel averages. This value is used to
        # normalize the equalization values, imposing that the mean of the distribution
        # of the equalization values is 1.0.
        # Note that we are using nanmean and masking values above zero to avoid
        # numerical issues.
        mean = np.nanmean(mu[mu > 0])
        mu /= mean
        # Write the equalization values to the calibration matrix.
        self.cal_matrix.values = np.where(mu > 0, mu, self.cal_matrix.values)
        # Update the entries.
        self.cal_matrix.entries = self._stats.counts()
        # Calculate the errors on the pixels as the standard deviation of the
        # sample mean, divided by the total mean.
        mask = self.cal_matrix.entries > 1
        denominator = mean * np.sqrt(self.cal_matrix.entries - 1, where=mask,
                              out=np.full_like(self.cal_matrix.entries, np.nan, dtype=float))
        np.divide(std, denominator, out=self.cal_matrix.errors, where=mask)
        self.cal_matrix.update_metadata(CalibrationMetadata.ADC_TO_EV, 1.)
        return self.cal_matrix

    def fit(self, **kwargs) -> CalibrationMatrix:
        """Calculate the equalization calibration matrix.

        Returns
        -------
        equalization_matrix : CalibrationMatrix
            Updated calibration matrix with the equalization values calculated from the data.
        """
        if self._algorithm == "absolute":
            return self._fit_absolute(**kwargs)
        if self._algorithm == "relative":
            return self._fit_relative()
        raise ValueError(f"Invalid algorithm {self._algorithm} for the equalization"
                          " calibration.")


class CalibrateGain:

    """Class for calibrating the gain of the readout chip.

    This class provides methods to calculate the gain values based on the equalization
    matrix and sensor material properties.
    """

    def __init__(self, equalization_matrix: CalibrationMatrix, material_symbol: str) -> None:
        """Class constructor.
        """
        num_rows, num_cols = equalization_matrix.shape
        self.cal_matrix = CalibrationMatrix(num_cols, num_rows)
        self.equalization_matrix = equalization_matrix
        self.material_symbol = material_symbol

    def fit(self) -> CalibrationMatrix:
        """Calculate the gain values based on the equalization matrix and the sensor material
        properties, and update the calibration matrix with the calculated values.
        """
        # Get the ionization potential of the sensor material and the conversion factor
        # from ADC to eV from the metadata of the equalization matrix.
        ionization_potential = xraydb.ionization_potential(self.material_symbol)
        adc_to_ev = self.equalization_matrix.metadata[CalibrationMetadata.ADC_TO_EV]
        # Calculate the conversion factor from ADC to electrons.
        adc_to_electrons = adc_to_ev / ionization_potential
        # Calculate the gain values as the equalization values divided by the conversion
        # factor and update the errors.
        self.cal_matrix.values = self.equalization_matrix.values / adc_to_electrons
        self.cal_matrix.errors = self.equalization_matrix.errors / adc_to_electrons
        # Update some metadata for the gain matrix.
        self.cal_matrix.entries = self.equalization_matrix.entries
        self.cal_matrix.num_events = self.equalization_matrix.num_events
        return self.cal_matrix


class CalibrateENC:

    """Class for calibrating the equivalent noise charge (ENC) of the readout chip.

    This class provides methods to calculate the ENC values based on the noise and gain
    matrices.
    """

    def __init__(self, noise_matrix: CalibrationMatrix, gain_matrix: CalibrationMatrix) -> None:
        """Class constructor.
        """
        if noise_matrix.shape != gain_matrix.shape:
            raise ValueError("Noise and gain matrices must have the same shape.")
        num_rows, num_cols = noise_matrix.shape
        self.cal_matrix = CalibrationMatrix(num_cols, num_rows)
        self.noise_matrix = noise_matrix
        self.gain_matrix = gain_matrix

    def fit(self) -> CalibrationMatrix:
        """Calculate the ENC values based on the noise and gain matrices, and update the
        calibration matrix with the calculated values.
        """
        # Calculate the ENC values as the ratio of the noise and gain values for each pixel.
        # If we have a NaN in a pixel, the result will be NaN. If we divide by zero, we set
        # the result to NaN as well.
        enc_values = np.divide(
            self.noise_matrix.values,
            self.gain_matrix.values,
            out=np.full_like(self.noise_matrix.values, np.nan),
            where=self.gain_matrix.values > 0
        )
        # Calculate the uncertainty with the same logic.
        rel_noise_sq = np.divide(self.noise_matrix.errors, self.noise_matrix.values,
                                out=np.zeros_like(self.noise_matrix.values),
                                where=self.noise_matrix.values > 0)**2

        rel_gain_sq = np.divide(self.gain_matrix.errors, self.gain_matrix.values,
                                out=np.zeros_like(self.gain_matrix.values),
                                where=self.gain_matrix.values > 0)**2
        enc_errors = enc_values * np.sqrt(rel_noise_sq + rel_gain_sq)
        # Update the calibration matrix with the calculated values.
        self.cal_matrix.values = enc_values
        self.cal_matrix.entries = np.minimum(self.noise_matrix.entries, self.gain_matrix.entries)
        self.cal_matrix.errors = enc_errors
        self.cal_matrix.num_events = self.noise_matrix.num_events
        return self.cal_matrix


class CalibratePosition:

    """Class to perform the calibration of the MLE position reconstruction algorithm. 
    
    This class implements the logic to create a set of calibration matrices to determine
    the fraction of charge collected by each pixel as a function of the incident position
    of the photon on the central pixel of the cluster. The calibration matrices are
    calculated by creating a grid of bins and calculating the average value of the
    fraction of charge collected by each pixel for each bin over all the events that fall
    in that bin.

    Arguments
    ---------
    bin_size: float
        The size of the square bins in the grid, in units of pixel pitch.
    num_bins: int
        The number of bins to use for the eta calibration fits.
    grid: HexagonalGrid
        The hexagonal grid with the same geometry of the detector.
    """

    PIXEL_SIZE = dict(
        flat_topped=(2 / np.sqrt(3), 1),
        pointy_topped=(1, 2 / np.sqrt(3))
    )

    def __init__(self, bin_size: float, num_bins: int, grid: HexagonalGrid) -> None:
        """Class constructor.
        """
        if not (0 < bin_size <= 0.25):
            raise ValueError(f"Invalid bin size: {bin_size}. Bin size must be between (0, 0.25].")
        self.bin_size = bin_size
        self._num_bins = num_bins
        self.grid = grid
        # Calculate the bin edges according to the pixel orientation.
        if grid.flat_topped():
            x_size, y_size = self.PIXEL_SIZE["flat_topped"]
        else:
            x_size, y_size = self.PIXEL_SIZE["pointy_topped"]
        # Calculate the bin edges.
        self._x_edges = np.arange(-x_size / 2, x_size / 2 + bin_size, bin_size)
        self._y_edges = np.arange(-y_size / 2, y_size / 2 + bin_size, bin_size)
        # Calculate the bin centers.
        self.x_bins = (self._x_edges[:-1] + self._x_edges[1:]) / 2
        self.y_bins = (self._y_edges[:-1] + self._y_edges[1:]) / 2
        # Create the lists to store the data for the eta calibration.
        self._position_2 = []
        self._eta_2 = []
        self._versors_2 = []
        self._position_3 = []
        self._eta_3 = []
        self._versors_3 = []
        # Create the PositionCalibrationData object to store the calibration data.
        self.cal_data = PositionCalibrationData(x_bins=self.x_bins, y_bins=self.y_bins)
        # Initialize the running statistics for the MLE calibration.
        self._stats = RunningStats(shape=self.cal_data.values.shape)

    def analyze_hex_cluster(self, cluster: Cluster, mc_event: MonteCarloEvent) -> None:
        """Analyze the event cluster and update the calibration data for the
        MLE calibration. The cluster must be obtained with the hexagonal clustering.

        Arguments
        ---------
        cluster : Cluster
            The cluster of pixels to analyze.
        mc_event : MonteCarloEvent
            The Monte Carlo event corresponding to the cluster.
        """
        # Calculate the Monte Carlo pixel normalized impact coordinates relative
        # to the central pixel of the cluster.
        x_rel = (mc_event.absx - cluster.x[0]) / self.grid.pitch
        y_rel = (mc_event.absy - cluster.y[0]) / self.grid.pitch
        # Find the corresponding bin in the data matrices using bin edges.
        # np.digitize returns the index i such that edges[i-1] <= x < edges[i].
        x_bin = np.digitize(x_rel, self._x_edges) - 1
        y_bin = np.digitize(y_rel, self._y_edges) - 1
        # Clamp indices to valid range to handle edge cases.
        x_bin = np.clip(x_bin, 0, len(self.x_bins) - 1)
        y_bin = np.clip(y_bin, 0, len(self.y_bins) - 1)
        # Calculate the charge fractions for the cluster.
        charge_fractions = cluster.pha / cluster.pulse_height()
        # Update the calibration data.
        # Maybe we can find a better way to use running stats here.
        for i in range(7):
            frac = np.array([[[charge_fractions[i]]]])
            self._stats.update(frac, offset=(i, x_bin, y_bin))

    def analyze_nn_cluster(self, cluster: Cluster, mc_event: MonteCarloEvent) -> None:
        """Analyze the event cluster and update the calibration data for the eta
        calibration. The cluster must be obtained with the nearest neighbor clustering.

        Arguments
        ---------
        cluster : Cluster
            The cluster of pixels to analyze.
        mc_event : MonteCarloEvent
            The Monte Carlo event corresponding to the cluster.
        """
        # Calculate the size of the cluster.
        size = cluster.size()
        # If the size is not 2 or 3, do nothing...
        if size not in [2, 3]:
            return
        # Otherwise calculate the coordinates of the Monte Carlo incident
        # position relative to the central pixel of the cluster, normalized
        # to the pixel pitch.
        x_rel = (mc_event.absx - cluster.x[0]) / self.grid.pitch
        y_rel = (mc_event.absy - cluster.y[0]) / self.grid.pitch
        # Calculate the eta values for the cluster.
        eta = cluster.pha[1:] / np.sum(cluster.pha)
        # Fill the lists for the calibration.
        if size == 2:
            self._eta_2.append(eta[0])
            self._position_2.append((x_rel, y_rel))
            self._versors_2.append(versor_2pix(cluster.x, cluster.y))
        elif size == 3:
            self._position_3.append((x_rel, y_rel))
            self._eta_3.append(eta)
            self._versors_3.append(versor_3pix(cluster.x, cluster.y))

    def _fit_size_2(self) -> None:
        """Fit the data for the 2-pixel clusters to determine the best-fit parameter.
        """
        # Convert to arrays
        self._eta_2 = np.array(self._eta_2)
        self._position_2 = np.array(self._position_2)
        self._versors_2 = np.array(self._versors_2)
        # Calculate the projected distance from the pixel center onto the 2-pixel
        # cluster versor.
        dr = np.sum(self._position_2 * self._versors_2, axis=1)
        # Calculate the profile of dr as a function of the eta values.
        x, y, yerr = profile(self._eta_2, dr, self._num_bins, 101)
        # Fit the profile with a probit model.
        model = Probit()
        model.offset.freeze(0.5)
        model.fit(x, y, sigma=yerr, absolute_sigma=True)
        # Update the calibration data with the fitted parameter.
        self.cal_data.two_pix_rad_sigma = model.sigma.value

    def _fit_size_3(self) -> None:
        """Fit the data for the 3-pixel clusters to determine the best-fit parameters.
        """
        # Convert to arrays
        self._eta_3 = np.array(self._eta_3)
        self._position_3 = np.array(self._position_3)
        self._versors_3 = np.array(self._versors_3)
        # Calculate the distance from the pixel center and the eta variable
        # for the radial fit.
        dr = np.sqrt(np.sum(self._position_3**2, axis=1))
        eta_sum = np.sum(self._eta_3, axis=1)
        # Calculate the profile of dr as a function of the eta sum.
        x, y, yerr = profile(eta_sum, dr, self._num_bins, 101)
        # Fit the profile with a probit model.
        model_r = Probit()
        model_r.fit(3 / 4 * x, y, sigma=yerr, absolute_sigma=True)
        # Calculate the angle of the incident position with respect to the versors
        # of the 3-pixel cluster.
        x_proj = np.sum(self._position_3 * self._versors_3[:, 0], axis=1)
        y_proj = np.sum(self._position_3 * self._versors_3[:, 1], axis=1)
        theta = np.arctan2(y_proj, x_proj)
        # Calculate the distance from the radial versor and the eta variable for
        # the angular fit.
        dy = dr * theta
        eta_diff = (self._eta_3[:, 0] - self._eta_3[:, 1]) / eta_sum
        # Calculate the profile of the distance from the radial versor as a
        # function of the eta difference.
        x, y, yerr = profile(eta_diff, dy, self._num_bins, 101)
        # Fit the profile with a probit model.
        model_theta = Probit()
        model_theta.offset.freeze(0)
        model_theta.fit((1 + x)/2, y, sigma=yerr, absolute_sigma=True)
        # Update the calibration data with the fitted parameters.
        self.cal_data.three_pix_rad_offset = model_r.offset.value
        self.cal_data.three_pix_rad_sigma = model_r.sigma.value
        self.cal_data.three_pix_theta_sigma = model_theta.sigma.value

    def fit(self) -> PositionCalibrationData:
        """Calculate the average charge fractions for each bin and store them in the
        MLECalibrationData object.

        Returns
        -------
        cal_data : PositionCalibrationData
            The PositionCalibrationData object containing the calibration data.
        """
        # Run the fits for the eta calibration.
        self._fit_size_2()
        self._fit_size_3()
        # Calculate the average charge fractions for the MLE calibration and
        # upfate the calibration data.
        self.cal_data.values = self._stats.mean()
        return self.cal_data
