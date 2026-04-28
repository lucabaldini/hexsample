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

from enum import Enum
from typing import Tuple

import h5py
import numpy as np
from aptapy.hist import Histogram3d
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr

from .clustering import Cluster
from .digi import DigiEventRectangular
from .recon import DEFAULT_IONIZATION_POTENTIAL


class CalibrationMetadata(str, Enum):

    """Enum to store the metadata keys for the calibration matrix.
    """

    NUM_COLS = "num_cols"
    NUM_ROWS = "num_rows"
    NUM_EVENTS = "num_events"
    NUM_EVENTS_AVG = "num_events_avg"
    NUM_EVENTS_MIN = "num_events_min"
    NUM_EVENTS_MAX = "num_events_max"
    NUM_CALIBRATED_PIXELS = "num_calibrated_pixels"
    VERSION = "version"
    KIND = "kind"
    IS_SYNTHETIC = "is_synthetic"


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
        self._num_events = 0
        self._metadata = dict(
            num_cols=num_cols,
            num_rows=num_rows
            )

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
            num_events_avg = int(self._entries[mask].mean())
            num_events_min = min(self._entries[mask])
            num_events_max = max(self._entries[mask])
        else:
            num_events_avg = num_events_min = num_events_max = 0
        # Setting the metadata values.
        self._metadata[CalibrationMetadata.NUM_EVENTS] = self._num_events
        self._metadata[CalibrationMetadata.NUM_EVENTS_AVG] = num_events_avg
        self._metadata[CalibrationMetadata.NUM_EVENTS_MIN] = num_events_min
        self._metadata[CalibrationMetadata.NUM_EVENTS_MAX] = num_events_max
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

    def to_hdf5(self, file_path: str, kind: str, is_synthetic: bool) -> str:
        """Save the calibration matrix to an HDF5 file at the given path.

        Arguments
        ---------
        file_path : str
            The path of the file on the disk.
        kind : str
            The kind of calibration for which the matrix is being saved.
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
            h5file.create_dataset(self.VALUES, data=self.values, **compression_pars)
            h5file.create_dataset(self.ENTRIES, data=self.entries, **compression_pars)
            h5file.create_dataset(self.ERRORS, data=self.errors, **compression_pars)
            # Update the header with the relevant information and metadata.
            self._metadata[CalibrationMetadata.KIND] = kind
            self._metadata[CalibrationMetadata.IS_SYNTHETIC] = is_synthetic
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
        if file_path is None:
            raise ValueError("No file path provided for the calibration matrix.")
        # pylint: disable=protected-access
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
        """Remove the signal pixels from the event pha array, by setting all the pixels in the 3x3
        region around the highest pixel to zero.

        Arguments
        ---------
        event : DigiEventRectangular
            The event to be analyzed.
        """
        seed_col, seed_row = event.highest_pixel(absolute=False)
        pha = event.pha.copy()
        pha[seed_row - 1: seed_row + 2, seed_col - 1: seed_col + 2] = 0
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
            self.cal_matrix._num_events += 1

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
        """Remove the signal pixels from the event pha array, by setting all the pixels in the 3x3
        region around the highest pixel to zero.

        Arguments
        ---------
        event : DigiEventRectangular
            The event to be analyzed.
        """
        seed_col, seed_row = event.highest_pixel(absolute=False)
        pha = event.pha.copy()
        pha[seed_row - 1: seed_row + 2, seed_col - 1: seed_col + 2] = 0
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

    def __init__(self, num_cols: int, num_rows: int, energy: float) -> None:
        """Class constructor.
        """
        super().__init__(num_cols, num_rows)
        self._energy = energy

        self._event_count = 0
        self._pha = []
        self._coords = []
        self._event_rows = []

    def analyze_cluster(self, cluster: Cluster) -> None:
        """Analyze the event cluster to update the calibration matrix.
        """
        # Get the coordinates of the cluster pixels
        cols = cluster.col
        rows = cluster.row
        # Update the arrays for the least squares fit.
        self._pha.extend(cluster.pha)
        for col, row in zip(cols, rows):
            # Calculate the index of the pixel in the flattened array
            i = row * self.cal_matrix.shape[1] + col
            self._coords.append(i)
            self._event_rows.append(self._event_count)
            # Update the matrix with the number of events for each pixel
            self.cal_matrix.entries[row, col] += 1
        # Update the event count
        self._event_count += 1
        self.cal_matrix._num_events += 1

    def fit(self) -> CalibrationMatrix:
        """Perform the least squares fit to determine the gain of each pixel.
        """
        if self._event_count == 0:
            raise ValueError("No events have been analyzed, cannot perform the fit.")
        nrows, ncols = self.cal_matrix.shape
        # Create the sparse matrix for the least squares fit. This object allows to store
        # and use efficiently the large and sparse matrix that we need for the fit.
        shape = (self._event_count, nrows * ncols)
        a = csr_matrix((self._pha, (self._event_rows, self._coords)), shape=shape)
        # Create the vector of the expected number of electrons.
        b = np.full(self._event_count, self._energy / DEFAULT_IONIZATION_POTENTIAL)
        # Perform the fit
        results = lsmr(a, b)
        # Get the best-fit weight vector and reshape it to the shape of the calibration matrix.
        weight = results[0].reshape((nrows, ncols))
        signal_power = np.array(a.multiply(a).sum(axis=0)).reshape((nrows, ncols))
        # Calculate the number of degrees of freedom and the mean squared error.
        dof = self._event_count - (ncols * nrows)
        mse = results[3]**2 / max(dof, 1)
        # Calculate the uncertainty of the gain best-fit values.
        sigma_w = np.sqrt(mse / (signal_power + 1e-15))
        with np.errstate(divide='ignore', invalid='ignore'):
            sigma_g_rel = sigma_w / np.abs(weight)
        # Mask for the pixels that have a weight value close to zero (no events) and for the pixels
        # with a large uncertainty. We are cutting out pixels with a relative uncertainty larger
        # than 200%. This value is high because the statistical fluctuations of the number of
        # electrons for each event affect the gain estimation, and even if the final gain
        # distribution is peaked around the true gain value, the uncertainty of the single pixel
        # can be large, even 100%.
        mask = (np.abs(weight) > 1e-10) & (sigma_g_rel < 2.0)
        values = self.cal_matrix.values.copy()
        entries = self.cal_matrix.entries
        # Set the gain value for the pixels that pass the quality cut.
        values[mask] = 1 / weight[mask]
        # Set the entries to zero for the pixels that don't pass the quality cut.
        entries[~mask] = 0
        # Write back through the setter so updates persist on the shared object.
        self.cal_matrix.values = values
        self.cal_matrix.errors = np.where(mask, sigma_g_rel * values, self.cal_matrix.errors)
        self.cal_matrix.entries = entries
        return self.cal_matrix
