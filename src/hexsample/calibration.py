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


from typing import Optional, Tuple

import h5py
import numpy as np
from aptapy.hist import Histogram2d
from aptapy.models import Probit
from aptapy.plotting import last_line_color, plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr

from ._version import __version__
from .clustering import Cluster
from .digi import DigiEventRectangular
from .recon import DEFAULT_IONIZATION_POTENTIAL


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

    def __init__(self, num_cols: int, num_rows: int) -> None:
        """Class constructor.
        """
        # pylint: disable=unused-argument
        self._shape = (num_rows, num_cols)
        # Create the arrays to store the calibration data and the number of events for each pixel.
        self._matrix = np.full(self._shape, np.nan)
        self._hits = np.zeros(self._shape, dtype=int)
        self._error = np.full(self._shape, np.nan)
        # Other useful information for the metadata
        self._num_events = 0
        self._feature = None
        self._is_synthetic = False

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the calibration matrix.
        """
        return self._shape

    @property
    def matrix(self) -> np.ndarray:
        """Return the calibration matrix.
        """
        return self._matrix

    @matrix.setter
    def matrix(self, new_matrix: np.ndarray) -> None:
        """Set the value of the calibration matrix to a new value.
        """
        # Check the consistency of the shape of the new matrix.
        if new_matrix.shape != self._shape:
            raise ValueError(f"Input matrix has shape {new_matrix.shape}, but expected shape is "
                             f"{self._shape}.")
        self._matrix = new_matrix

    @property
    def hits(self) -> np.ndarray:
        """Return the number of events for each pixel in the calibration matrix.
        """
        return self._hits

    @property
    def error(self) -> np.ndarray:
        """Return the error of the calibration matrix for each pixel.
        """
        return self._error

    @error.setter
    def error(self, new_error: np.ndarray) -> None:
        """Set the value of the error of the calibration matrix to a new value.
        """
        # Check the consistency of the shape of the new error matrix.
        if new_error.shape != self._shape:
            raise ValueError(f"Input error matrix has shape {new_error.shape}, but expected shape "
                             f"is {self._shape}.")
        self._error = new_error

    @property
    def metadata(self) -> dict:
        """Return the metadata of the calibration matrix.
        """
        mask = self._hits > 0
        _metadata = dict(
            num_cols=self._shape[1],
            num_rows=self._shape[0],
            num_events=self._num_events,
            num_events_avg=int(self._hits[mask].mean()),
            num_events_min=min(self._hits[mask]),
            num_events_max=max(self._hits[mask]),
            num_calibrated_pixels=mask.sum(),
            version=__version__,
            feature=self._feature,
            is_synthetic=self._is_synthetic
        )
        return _metadata

    def set_value(self, value: float) -> None:
        """Set a value for all the pixels in the calibration matrix.

        Arguments
        ---------
        value : float
            The value to be set for all the pixels in the calibration matrix.
        """
        self._matrix = np.full(self._shape, value)

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
        self._matrix = np.where(self._hits <= max_hits, value, self._matrix)

    def mean(self, min_hits: int = 1) -> float:
        """Return the mean value of the calibration matrix, calculated as the mean of the pixels
        with at least one event.

        Arguments
        ---------
        min_hits : int
            The minimum number of hits for a pixel to be considered for the mean calculation.
        """
        if not np.any(self._hits >= min_hits):
            return np.nan
        return self._matrix[self._hits >= min_hits].mean()

    def to_hdf5(self, file_path: str, feature: str, is_synthetic: bool) -> str:
        """Save the calibration matrix to an HDF5 file at the given path.

        Arguments
        ---------
        file_path : str
            The path of the file on the disk.
        feature : str
            The feature for which the calibration matrix is being saved.
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
            h5file.create_dataset("matrix", data=self.matrix, **compression_pars)
            h5file.create_dataset("hits", data=self.hits, **compression_pars)
            h5file.create_dataset("error", data=self.error, **compression_pars)
            # Update the header with the relevant information and metadata.
            self._feature = feature
            self._is_synthetic = is_synthetic
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
                setattr(obj, f"_{key}", val)
            # Load the matrix and the hits matrices from the HDF5 file.
            obj._matrix = h5file["matrix"][:]
            obj._hits = h5file["hits"][:]
            obj._error = h5file["error"][:]
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
        return self.matrix[row, col]


class CalibrateNoise:

    """Calibrate the noise of the detector by analyzing the events in a DigiFile. This class
    takes a CalibrationMatrix object as input, and updates the matrix with the data from
    the file.

    In principle, the dataset used for this task should contain data uniformly distributed
    across the detector.

    Arguments
    ---------
    cal_matrix : CalibrationMatrix
        Calibration matrix to be updated with the noise values calculated from the data.
    """

    def __init__(self, cal_matrix: CalibrationMatrix) -> None:
        """Class constructor.
        """
        self.cal_matrix = cal_matrix
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
            self.cal_matrix.hits[row_slice, col_slice][noise_pha > 0] += 1
            self.cal_matrix._num_events += 1

    def update(self):
        """Update the calibration matrix with the noise values calculated from the data.
        """
        matrix = self.cal_matrix.matrix.copy()
        hits = self.cal_matrix.hits
        # If the sum2 array is still zero, it means that no events have been analyzed, so we can
        # set the matrix to the default value for all the pixels.
        if np.array_equal(self._sum2, np.zeros(self.cal_matrix.shape)):
            raise ValueError("No events have been analyzed, cannot update the calibration matrix.")
        with np.errstate(divide='ignore', invalid='ignore'):
            matrix = np.where(hits > 0, np.sqrt(self._sum2 / hits), matrix)
            error = np.where(hits > 1, matrix / np.sqrt(2 * (hits - 1)), self.cal_matrix.error)
        # Write back through the setter so updates persist on the shared object.
        self.cal_matrix.matrix = matrix
        self.cal_matrix.error = error


class CalibrateGain:

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

    def __init__(self, calibration_matrix: CalibrationMatrix, energy: float) -> None:
        """Class constructor.
        """
        self.cal_matrix = calibration_matrix
        self._shape = calibration_matrix.shape
        self._energy = energy

        self._event_count = 0
        self._pha = []
        self._coords = []
        self._event_rows = []

    def fit(self) -> None:
        """Perform the least squares fit to determine the gain of each pixel.
        """
        if self._event_count == 0:
            raise ValueError("No events have been analyzed, cannot perform the fit.")
        nrows, ncols = self._shape
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
        matrix = self.cal_matrix.matrix.copy()
        hits = self.cal_matrix.hits
        # Set the gain value for the pixels that pass the quality cut.
        matrix[mask] = 1 / weight[mask]
        # Set the hits to zero for the pixels that don't pass the quality cut.
        hits[~mask] = 0
        # Write back through the setter so updates persist on the shared object.
        self.cal_matrix.matrix = matrix
        self.cal_matrix.error = np.where(mask, sigma_g_rel * matrix, self.cal_matrix.error)

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
            i = row * self._shape[1] + col
            self._coords.append(i)
            self._event_rows.append(self._event_count)
            # Update the matrix with the number of events for each pixel
            self.cal_matrix.hits[row, col] += 1
        # Update the event count
        self._event_count += 1
        self.cal_matrix._num_events += 1


def profile(xdata: np.ndarray, ydata: np.ndarray, xbins: int, ybins: int
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the profile of a set of xdata and ydata. The profile is computed by creating
    the 2D histogram and computing the median of the y-axis distribution for each x-bin.

    Arguments
    ---------
    xdata : np.ndarray
        The x data to be profiled.
    ydata : np.ndarray
        The y data to be profiled.
    xbins : int
        The number of bins in the x axis.
    ybins : int
        The number of bins in the y axis.

    Returns
    -------
    x : np.ndarray
        The bin centers in the x axis.
    y : np.ndarray
        The median values in the y axis for each x bin.
    yerr: np.ndarray
        The errors of the median values in the y axis for each x bin.
    """
    # Create the 2D histogram to compute the profile
    xedges = np.linspace(xdata.min(), xdata.max(), xbins + 1).flatten()
    yedges = np.linspace(ydata.min(), ydata.max(), ybins + 1).flatten()
    hist = Histogram2d(xedges, yedges)
    hist.fill(xdata, ydata)
    # Create the arrays to store the profile values and their errors
    x = hist.bin_centers()
    y = np.zeros(xbins)
    yerr = np.zeros(xbins)
    for i in range(xbins):
        # Slice the histogram on the x-axis to get the y-axis distribution
        hist_slice = hist.slice1d(i)
        entries = hist_slice.content.sum()
        if entries == 0:
            # If the slice histogram is empty, set the profile value and error to NaN
            y[i] = np.nan
            yerr[i] = np.nan
            continue
        # Compute the profile value as the median of the y-axis distribution
        y[i] = hist_slice.ppf(0.5)
        # Compute the error of the sample median
        yerr[i] = 1.253 * hist_slice.binned_statistics()[1] / np.sqrt(entries)
    # Remove the bins with NaN values and return the profile
    valid = ~np.isnan(y)
    return x[valid], y[valid], yerr[valid]


def angle(pos: np.ndarray, versors: np.ndarray) -> np.ndarray:
    """Calculate the angle between the photon position and the versors of the cluster.

    Arguments
    ---------
    pos : np.ndarray
        The position of the photon with respect to the most charged pixel, in units of pitch.
    versors : np.ndarray
        The versors of the cluster.

    Returns
    -------
    angle : np.ndarray
        The angle between the photon position and the versors of the cluster, in radians.
    """
    # Calculate the projections of the photon position on the versors
    x_proj = np.sum(pos * versors[:, 0], axis=1)
    y_proj = np.sum(pos * versors[:, 1], axis=1)
    # Estimate the angle
    return np.arctan2(y_proj, x_proj)


def distance(pos: np.ndarray, projection_axis: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate the distance of the photon from the center of the most charged pixel. If
    specified, project the distance on the given projection axis, given as a unit vector.

    Arguments
    ---------
    pos : np.ndarray
        The position of the photon with respect to the most charged pixel, in units of pitch.
    projection_axis : np.ndarray, optional
        The axis on which to project the distance, given as a unit vector. If None,
        the distance is not projected. Default is None.

    Returns
    -------
    distance : np.ndarray
        The distance of the photon from the center of the most charged pixel. If a projection axis
        is specified, the distance is projected onto that axis.
    """
    if projection_axis is None:
        # This is useful for 3-pixel events
        return np.sqrt(np.sum(pos**2, axis=1))
    # This is useful for 2-pixel events
    return np.sum(pos * projection_axis, axis=1)


def calibrate_dr_2pix(eta: np.ndarray, dr: np.ndarray, nbins: int, **kwargs) -> float:
    """Calibrate the 2-pixel eta function using the distance projection on the line connecting
    the two highest pixels in the cluster. The eta function is fitted with a probit model with the
    offset fixed to 0.5.

    Arguments
    ---------
    eta : np.ndarray
        The eta values for the 2-pixel clusters.
    dr : np.ndarray
        The distance of the photon from the center of the most charged pixel, projected onto the
        line connecting the two pixels, in units of pitch.

    Returns
    -------
    sigma : float
        The best-fit value of the sigma parameter of the probit function.
    """
    # Calculate the profile of the data
    x, y, yerr = profile(eta, dr, nbins, 101)
    # Fit the data with a probit model with offset fixed to 0.5
    model = Probit()
    model.offset.freeze(0.5)
    model.fit(x, y, sigma=yerr, absolute_sigma=True)
    # Plot the results
    fig = plt.figure("calibration_eta_vs_dr_2pix")
    plt.errorbar(x, y, yerr=yerr, fmt=".k", label="Monte Carlo simulation")
    fit_label = "2-pixel events calibration\n" + fr"$\sigma$ = {model.sigma.ufloat()}"
    model.plot(label=fit_label, color=last_line_color())
    plt.xlabel(r"$\eta$")
    plt.ylabel("r/p")
    plt.legend()
    # Save the figure if requested
    if kwargs.get("save", False):
        fig_path = kwargs.get("path")
        fig.savefig(fig_path / "2pix_cal.pdf", format="pdf")
    # Test with a pivot
    eta_pivot = 0.0423
    xx = np.linspace(0, max(x), 100)
    yy = np.where(xx < eta_pivot, model(eta_pivot) / eta_pivot * xx, model(xx))
    plt.figure("test fit_2pix")
    plt.plot(xx, yy, label="Fitted model", color=last_line_color())
    plt.errorbar(x, y, yerr=yerr, fmt=".k", label="Monte Carlo simulation")
    return model.sigma.value


def calibrate_dr_3pix(eta: np.ndarray, dr: np.ndarray, nbins: int, **kwargs) -> Tuple[float, float]:
    """Calibrate the radial component of the 3-pixel eta function, using the distance of the photon
    from the center of the most charged pixel. The eta function is fitted with a probit model.

    Arguments
    ---------
    eta : np.ndarray
        The eta values for the 3-pixel clusters.
    dr : np.ndarray
        The distance of the photon from the center of the most charged pixel, projected onto the
        line connecting the two pixels, in units of pitch.

    Returns
    -------
    offset : float
        The best-fit value of the offset parameter of the probit function.
    sigma : float
        The best-fit value of the sigma parameter of the probit function.
    """
    # Calculate the sum of the eta values for each event
    eta_sum = np.sum(eta, axis=1) * 3 / 4
    # Calculate the profile of the data
    x, y, yerr = profile(eta_sum, dr, nbins, 101)
    # Fit with a probit model
    model = Probit()
    model.fit(x, y, sigma=yerr, absolute_sigma=True)
    # Plot the results
    fig = plt.figure("calibration_eta_vs_dr_3pix")
    plt.errorbar(x, y, yerr=yerr, fmt=".k", label="Monte Carlo simulation")
    fit_label = "3-pixel events radial calibration\n" + fr"$\sigma$ = {model.sigma.ufloat()}"
    fit_label += "\n" + fr"$\mu$ = {model.offset.ufloat()}"
    model.plot(label=fit_label, color=last_line_color())
    plt.xlabel(r"$\eta^+$")
    plt.ylabel("r/p")
    plt.legend()
    # Save the figure if requested
    if kwargs.get("save", False):
        fig_path = kwargs.get("path")
        fig.savefig(fig_path / "3pix_cal_radial.pdf", format="pdf")
    # Test with a pivot
    eta_pivot = 0.054
    xx = np.linspace(0, max(x), 100)
    yy = np.where(xx < eta_pivot, model(eta_pivot) / eta_pivot * xx, model(xx))
    plt.figure("test fit_3pix")
    plt.plot(xx, yy, label="Fitted model", color=last_line_color())
    plt.errorbar(x, y, yerr=yerr, fmt=".k", label="Monte Carlo simulation")
    return model.offset.value, model.sigma.value


def calibrate_theta_3pix(eta: np.ndarray, dr: np.ndarray, theta: np.ndarray, nbins: int,
                         **kwargs) -> float:
    """Calibrate the angular component of the 3-pixel eta function using the angle between the
    photon position and the versor pointing from the center of the most charged pixel to the
    midpoint between the two less charged pixels, and the radial distance from the center of the
    most charged pixel. The eta function is fitted with a probit model with the offset fixed to 0.

    Arguments
    ---------
    eta : np.ndarray
        The eta values for the 3-pixel clusters.
    dr : np.ndarray
        The distance of the photon from the center of the most charged pixel, projected onto the
        line connecting the two pixels, in units of pitch.
    theta : np.ndarray
        The angle between the photon position and the versors of the cluster, in radians.

    Returns
    -------
    sigma : float
        The best-fit value of the sigma parameter of the probit function.
    """
    # Calculate the transverse component
    delta_r = dr * theta
    # Calculate eta-
    eta_diff = (eta[:, 0] - eta[:, 1]) / np.sum(eta, axis=1)
    # Calculate the profile of the data
    x, y, yerr = profile(eta_diff, delta_r, nbins, 101)
    # Fit with a probit model with offset fixed to 0.
    model = Probit()
    model.offset.freeze(0.)
    # We have to traslate the data to match the model definition.
    model.fit((1 + x)/2, y, sigma=yerr, absolute_sigma=True)
    # Plot the results
    fig = plt.figure("calibration_eta_vs_theta_3pix")
    plt.errorbar(x, y, yerr=yerr, fmt=".k", label="Monte Carlo simulation")
    fit_label = "3-pixel events angular calibration\n"
    fit_label += fr"$\sigma$ = {model.sigma.ufloat()}"
    # Plot the model
    xx = np.linspace(0.5, max((1 + x)/2), 100)
    plt.plot(2*xx - 1, model(xx), label=fit_label, color=last_line_color())
    plt.xlabel(r"$\eta^-$")
    plt.ylabel(r"r$\theta$/p")
    plt.legend()
    # Save the figure if requested
    if kwargs.get("save", False):
        fig.savefig(kwargs.get("path") / "3pix_cal_angular.pdf", format="pdf")
    return model.sigma.value
