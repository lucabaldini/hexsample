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
import pathlib
from enum import Enum
from itertools import product
from typing import Optional, Tuple

import h5py
import numpy as np
from aptapy.hist import Histogram3d
from iminuit import Minuit
from joblib import Parallel, delayed
from scipy.sparse import csc_matrix, csr_matrix
from tqdm import tqdm

from .clustering import Cluster
from .digi import DigiEventRectangular
from .pdf import SpectrumPDF


class CalibrationType(str, Enum):

    """Enum class expressing the possible calibration types.
    """

    ENC = "enc"
    PEDESTAL = "pedestal"
    NOISE = "noise"
    GAIN = "gain"

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


class CalibrationUnits(str, Enum):

    """Enum to store the possible units for the calibration matrix values.
    """

    ENC = "Electrons"
    NOISE = "ADC counts"
    PEDESTAL = "ADC counts"
    GAIN = ""


CALIBRATION_UNITS = {
    CalibrationType.ENC: CalibrationUnits.ENC,
    CalibrationType.NOISE: CalibrationUnits.NOISE,
    CalibrationType.PEDESTAL: CalibrationUnits.PEDESTAL,
    CalibrationType.GAIN: CalibrationUnits.GAIN
}


class CalibrationMatrix:

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

    VALUES = "values"
    ENTRIES = "entries"
    ERRORS = "errors"

    def __init__(self, num_cols: int, num_rows: int) -> None:
        """Class constructor.
        """
        # pylint: disable=unused-argument
        self._shape = (num_rows, num_cols)
        # Create the arrays to store the calibration data and the number of events for each pixel.
        self._values = np.full(self._shape, np.nan)
        self._entries = np.zeros(self._shape, dtype=int)
        self._errors = np.full(self._shape, np.nan)
        # Other useful information for the metadata
        self.num_events = 0
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
    def values(self) -> np.ndarray:
        """Return the calibration matrix.
        """
        return self._values

    @values.setter
    def values(self, new_matrix: np.ndarray) -> None:
        """Set the value of the calibration matrix to a new value.
        """
        # Check the consistency of the shape of the new matrix.
        if new_matrix.shape != self._shape:
            raise ValueError(f"Input matrix has shape {new_matrix.shape}, but expected shape is "
                             f"{self._shape}.")
        self._values = new_matrix

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

    @property
    def metadata(self) -> dict:
        """Return the metadata of the calibration matrix.
        """
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
        return self._metadata

    def set_value(self, value: float) -> None:
        """Set a value for all the pixels in the calibration matrix.

        Arguments
        ---------
        value : float
            The value to be set for all the pixels in the calibration matrix.
        """
        self._values = np.full(self._shape, value)

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
        self._values = np.where(self._entries <= max_hits, value, self._values)

    def mean(self, min_hits: int = 1) -> float:
        """Return the mean value of the calibration matrix, calculated as the mean of the pixels
        with at least one event.

        Arguments
        ---------
        min_hits : int
            The minimum number of hits for a pixel to be considered for the mean calculation.
        """
        if not np.any(self._entries >= min_hits):
            return np.nan
        return self._values[self._entries >= min_hits].mean()

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
        # pylint: disable=protected-access
        compression_pars = dict(
            compression="gzip",
            compression_opts=9,
            shuffle=True
        )
        with h5py.File(file_path, "w") as h5file:
            # Save the matrix and the hits matrices as arrays in the HDF5 file.
            h5file.create_dataset(self.VALUES, data=self.values, dtype=np.float32,
                                  **compression_pars)
            h5file.create_dataset(self.ENTRIES, data=self.entries, dtype=np.int32,
                                  **compression_pars)
            h5file.create_dataset(self.ERRORS, data=self.errors, dtype=np.float32,
                                  **compression_pars)
            # Update the header with the relevant information and metadata.
            self._metadata[CalibrationMetadata.CALIBRATION_TYPE] = calibration_type.value
            self._metadata[CalibrationMetadata.IS_SYNTHETIC] = is_synthetic
            self._metadata[CalibrationMetadata.FILE_NAME] = pathlib.Path(file_path).stem
            for key, val in self.metadata.items():
                h5file.attrs[key] = val
        return file_path

    @classmethod
    def from_hdf5(cls, file_path: str) -> "CalibrationMatrix":
        """Create an instance of the calibration matrix from an HDF5 file at the given path.

        Arguments
        ---------
        file_path : str
            The path of the file on the disk.
        """
        # pylint: disable=protected-access
        if file_path is None:
            raise ValueError("No file path provided for the calibration matrix.")
        # Check if the file exists before trying to open it.
        if not pathlib.Path(file_path).is_file():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with h5py.File(file_path, "r") as h5file:
            # Load the attributes from the header.
            attrs = dict(h5file.attrs)
            # Instantiate the object with the attributes loaded from the header.
            obj = cls(num_cols=attrs["num_cols"], num_rows=attrs["num_rows"])
            for key, val in attrs.items():
                obj._metadata[key] = val
            # Load the matrix and the hits matrices from the HDF5 file.
            obj._values = h5file[obj.VALUES][:]
            obj._entries = h5file[obj.ENTRIES][:]
            obj._errors = h5file[obj.ERRORS][:]
        return obj

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

    def __str__(self) -> str:
        """Return a string representation of the calibration matrix.
        """
        if CalibrationMetadata.FILE_NAME in self._metadata:
            return self._metadata[CalibrationMetadata.FILE_NAME]
        return f"CalibrationMatrix(num_cols={self._metadata[CalibrationMetadata.NUM_COLS]}, " \
                f"num_rows={self._metadata[CalibrationMetadata.NUM_ROWS]})"


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

    The calibration is performed by estimating the mean and the standard deviation of the pixel
    value distribution for each pixel, and using these values as the pedestal and noise values.

    Arguments
    ---------
    noise_matrix : CalibrationMatrix
        Calibration matrix to be updated with the noise values calculated from the data.
    pedestal_matrix : CalibrationMatrix
        Calibration matrix to be updated with the pedestal values calculated from the data.
    """

    def __init__(self, num_cols: int, num_rows: int) -> None:
        """Class constructor.
        """
        self.noise_cal = CalibrationMatrix(num_cols, num_rows)
        self.pedestal_cal = CalibrationMatrix(num_cols, num_rows)
        # Check if the noise and pedestal calibration matrices have the same shape.
        num_rows, num_cols = self.noise_cal.shape
        xedges = np.linspace(0, num_cols, num_cols + 1)
        yedges = np.linspace(0, num_rows, num_rows + 1)
        # For now just use a fixed number, but we need to fix this
        zedges = np.linspace(0, 2048, 2049)
        self._histogram = Histogram3d(xedges, yedges, zedges)
        # Batch analysis
        self._pha = []
        self._cols = []
        self._rows = []

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

    def update_hist(self) -> None:
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
        if self._bad_event(event):
            return
        pha = self._remove_signal(event) if has_source else event.pha
        # Find the coordinates of the pixels with pha > 0 in the event.
        local_rows, local_cols = np.nonzero(pha > 0)
        pha_values = pha[local_rows, local_cols]
        # Traslate the local coordinates to global coordinates.
        row_slice, col_slice = event.roi.readout_slice()
        global_rows = local_rows + row_slice.start
        global_cols = local_cols + col_slice.start
        # Accumulate the data to fill the histogram in batch.
        self._pha.extend(pha_values)
        self._cols.extend(global_cols)
        self._rows.extend(global_rows)
        self.noise_cal.num_events += 1
        self.pedestal_cal.num_events += 1
        # If the size of the accumulated data is large enough, fill the histogram.
        if len(self._pha) >= batch_size:
            self.update_hist()

    # def fit(self) -> None:
    #     noise = self.noise_cal.matrix.copy()
    #     pedestal = self.pedestal_cal.matrix.copy()
    #     model = Gaussian()
    #     print("Fitting noise and pedestal for each pixel...")
    #     # Should try to think about a more efficient way to perform 10^5 fits
    #     for col in range(self.noise_cal.shape[1]):
    #         print(f"Fitting column {col}...")
    #         for row in range(self.noise_cal.shape[0]):
    #             slice_ = self._histogram.slice1d(col, row)
    #             entries = slice_.content.sum()
    #             if entries > 0:
    #                 model.fit(slice_)
    #                 noise[row, col] = model.sigma
    #                 pedestal[row, col] = model.mu

    def fit(self) -> Tuple[CalibrationMatrix, CalibrationMatrix]:
        """Analyze the histogram to calculate the noise and pedestal values for each pixel, and
        update the calibration matrices.

        At the moment, the pedestal and noise values are estimated as the mean and the standard
        deviation of the pixel value distribution for each pixel.

        Returns
        -------
        noise_cal : CalibrationMatrix
             Updated calibration matrices for the noise.
        pedestal_cal : CalibrationMatrix
             Updated calibration matrices for the pedestal.
        """
        # Calculate the mean and the standard deviation of the pixel value distribution for
        # each pixel.
        histo_mean, histo_sigma = self._histogram.project_statistics(axis=2)
        mu = histo_mean.content.T
        sigma = histo_sigma.content.T
        # Update the noise and pedestal matrices with the calculated values for the pixels that
        # have at least one hit.
        noise_matrix = np.where(self.noise_cal.entries > 0, sigma, self.noise_cal.values)
        pedestal_matrix = np.where(self.pedestal_cal.entries > 0, mu, self.pedestal_cal.values)
        # Write the matrices
        self.noise_cal.values = noise_matrix
        mask = self.noise_cal.entries > 1
        # The error on the estimate of the standard deviation is given by sigma / sqrt(2 * (N - 1))
        self.noise_cal.errors[mask] = sigma[mask] / np.sqrt(2 * (self.noise_cal.entries[mask] - 1))
        self.pedestal_cal.values = pedestal_matrix
        # The error on the estimate of the mean is given by sigma / sqrt(N - 1)
        self.pedestal_cal.errors[mask] = sigma[mask] / np.sqrt(self.pedestal_cal.entries[mask] - 1)
        return self.noise_cal, self.pedestal_cal


def _likelihood_fit(data: csr_matrix, conv_factor: float, pdf: SpectrumPDF,
                    pdf_derivative: callable) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Perform the likelihood fit for a region of the detector.

    Returns
    -------
    gain : np.ndarray
        The gain values for the pixels in the region, calculated from the fit.
    error : np.ndarray
        The error on the gain values for the pixels in the region, calculated from the fit.
    """
    # Define the negative log-likelihood function for the fit.
    def nll(pars):
        total_adc = data @ pars
        p = pdf(total_adc * conv_factor)
        return -np.sum(np.log(p + 1e-10))
    # Define the gradient of the log-likelihood function for the fit.
    def nll_grad(pars):
        total_adc = data @ pars
        p = pdf(total_adc * conv_factor)
        dp = pdf_derivative(total_adc * conv_factor)
        grad = -data.T @ (dp / (p + 1e-10)) * conv_factor
        return np.asarray(grad).flatten()
    # Define the initial parameters for the fit.
    init_pars = np.ones(data.shape[1])
    # Initialize the Minuit minimizer.
    m = Minuit(nll, init_pars, grad=nll_grad)
    m.limits = [(1e-10, None) for _ in range(len(init_pars))]
    m.errordef = Minuit.LIKELIHOOD
    m.migrad()
    # If the fit is successful, return the gain values and their errors, otherwise
    # return None.
    if m.valid:
        gain = 1 / np.array(m.values)
        error = m.errors / np.array(m.values)**2
        return gain, error
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


def worker_fit_block(r_start: int, c_start: int, size: int, nrows: int, ncols: int,
                     data: csc_matrix, event_sum: np.ndarray, conv_factor: float,
                     pdf: SpectrumPDF, pdf_derivative: callable
                     ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Method to perform the fit for a block of pixels in the detector.
    This method is meant to be used as a function to be executed in parallel for different
    blocks of pixels.

    .. note:: This function must be defined outside the class to be picklable and usable
            in parallel execution.
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
    # Check on the data: if there are no good events or no active pixels,
    # we cannot perform the fit, so we return None.
    if data_fit.nnz == 0 or data_fit.shape[1] == 0:
        return None
    # Fit the data of the block.
    result = _likelihood_fit(data_fit, conv_factor, pdf, pdf_derivative)
    # If the fit is not successful, return None and continue with the next block.
    if result is None:
        return None
    # If the fit is successful, return the gain values and their errors and the
    # coordinates of the active pixels.
    return result[0], result[1], rows[active_mask], cols[active_mask]


class CalibrateGain(CalibrateBase):

    """Calibrate the gain of the detector by analyzing the events in a DigiFile. This class takes a
    CalibrationMatrix object as input, and updates the matrix with the data from the file.

    At the moment, the dataset used for this task should contain events from a monochromatic source
    and with known energy.

    Arguments
    ---------
    cal_matrix : CalibrationMatrix
        Calibration matrix to be updated with the gain values calculated from the data.
    energy : float
        The energy of the monochromatic source, in eV.
    """

    def __init__(self, num_cols: int, num_rows: int, pdf: SpectrumPDF) -> None:
        """Class constructor.
        """
        super().__init__(num_cols, num_rows)
        self._event_count = 0
        self._pha = []
        self._coords = []
        self._event_rows = []
        self._pdf = pdf
        self._pdf_derivative = pdf.derivative

    def analyze_cluster(self, cluster: Cluster) -> None:
        """Analyze the event cluster to update the calibration matrix.
        """
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
            # Update the matrix with the number of events for each pixel
            self.cal_matrix.entries[row, col] += 1
        # Update the event count
        self._event_count += 1
        self.cal_matrix.num_events += 1

    def fit(self) -> CalibrationMatrix:
        """Fit the collected events to determine the gain of each pixel.

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
        # Calculate the mean of the ADC counts in an event.
        conv_factor = self._pdf.mean() / event_sum.mean()
        # Define the size of the blocks to fit contemporarily.
        size = 6
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
        # Convert the CSR matrix to CSC format to optimize column slicing during data cuts.
        data_csc = data_csr.tocsc()
        # Free the memory used by the CSR matrix and temporary arrays.
        del data_csr, data_pha, data_coords, data_event_rows
        gc.collect()
        # Start the parallel processing of the blocks.
        args = (size, nrows, ncols, data_csc, event_sum, conv_factor,
                self._pdf, self._pdf_derivative)
        bar_format = "{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(worker_fit_block)(
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
            gain, error, rows, cols = _result
            weights = 1 / (error**2 + 1e-10)
            # Use the mask of the active pixels to update the weighted gains and the sum
            # of the weights matrices for the pixels in the block that have been fitted.
            np.add.at(weighted_gains, (rows, cols), gain * weights)
            np.add.at(sum_weights, (rows, cols), weights)
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
        return self.cal_matrix


class CalibrateENC:
    """Class for calibrating the equivalent noise charge (ENC) of the readout chip.

    This class provides methods to calculate the ENC values based on the noise and gain
    matrices.
    """

    def __init__(self, noise_matrix: CalibrationMatrix, gain_matrix: CalibrationMatrix) -> None:
        """Class constructor
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
