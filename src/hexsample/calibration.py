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

import numpy as np
import tables
from aptapy.hist import Histogram1d, Histogram2d
from aptapy.models import Gaussian, Probit
from aptapy.plotting import last_line_color, plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr
from tables.attributeset import AttributeSet

from .clustering import Cluster
from .digi import DigiEventRectangular
from .fileio import DigiInputFileBase
from .readout import HexagonalReadoutBase
from .recon import DEFAULT_IONIZATION_POTENTIAL


class CalibrationMatrixBase:

    """Base class for calibration analysis.

    This includes the common structure of the detector calibration matrices, which are used to
    calibrate each pixel of the detector, and the facilities to save and load calibration data
    from HDF5 files.

    The derived classes need to implement the logic to update the calibration matrix with info
    from the digitized events or the clusters.

    Arguments
    ---------
    num_cols : int
        The number of columns of the readout chip.
    num_rows : int
        The number of rows of the readout chip.
    default : float, optional
        The default value to set for pixels in the calibration matrix.
    """

    def __init__(self, num_cols: int, num_rows: int, default: Optional[float] = None,
                 **kwargs) -> None:
        """Class constructor.
        """
        # pylint: disable=unused-argument
        self._shape = (num_rows, num_cols)
        self._default = default
        # Create the arrays to store the calibration data and the number of events for each pixel.
        self._matrix = np.zeros(self._shape)
        self._hits = np.zeros(self._shape, dtype=int)
        self._sum = np.zeros(self._shape)

    @property
    def default(self) -> float:
        """Calculate the default value of the calibration matrix for pixels with no events. It is
        estimated as the mean of the values in the calibration matrix for pixels with events, in
        case no default value is provided.
        """
        # If the default value is provided, return it.
        if self._default is not None:
            return self._default
        # If the default value is not provided, but the calibration matrix has no events, return 0.
        if not np.any(self._hits > 0):
            return 0.
        # If the _sum array has been updated with data, calculate the default value as the mean of
        # the values for pixels with events.
        if not np.array_equal(self._sum, np.zeros(self._shape)):
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.mean(self._sum[self._hits > 0] / self._hits[self._hits > 0])
        # Otherwise, if the _matrix array has been updated directly, the default value is estimated
        # as the mean of the values for pixels with events in the _matrix array.
        return np.mean(self._matrix[self._hits > 0])

    @property
    def matrix(self) -> np.ndarray:
        """Return the calibration matrix.

        The way the calibration matrix is calculated from the data depends on the type of
        calibration and on the analysis method used.
        """
        # If the analysis updates the _sum and _hits arrays, calculate the value of eacg pixel
        # as the mean of the values for all the events for that pixel. For pixels with no events,
        # the default value is used.
        if not np.array_equal(self._sum, np.zeros(self._shape)):
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(self._hits > 0, self._sum / self._hits, self.default)
        # If no events have been used to update the calibration matrix, return the default value
        # for all the pixels.
        if not np.any(self._hits > 0):
            return np.full(self._shape, self.default)
        # Otherwise, if the calibration matrix is updated directly during the analysis, return the
        # value of the matrix for pixels with events, and the default value for pixels with no
        # events.
        return np.where(self._hits > 0, self._matrix, self.default)

    @matrix.setter
    def matrix(self, new_matrix: np.ndarray) -> None:
        """Set the value of the calibration matrix to a new value.
        """
        if new_matrix.shape != self._shape:
            raise ValueError(f"Input matrix has shape {new_matrix.shape}, but expected shape is "
                             f"{self._shape}.")
        self._matrix = new_matrix
        # Setting the hits to one to avoid that the default value is estimated from the data.
        self._hits = np.ones(self._shape, dtype=int)

    @property
    def hits(self) -> np.ndarray:
        """Return the number of events for each pixel in the calibration matrix.
        """
        return self._hits

    def _update_header(self, attrs: AttributeSet):
        """Update the header of the HDF5 file with the relevant information for the calibration
        matrix.
        """

    @staticmethod
    def _load_header_dict(attrs: AttributeSet) -> dict:
        """Load the header of the HDF5 file and return a dictionary with the relevant attributes
        for the calibration matrix.
        """
        # pylint: disable=protected-access
        # Load all the attributes from the header.
        header_dict = {name: getattr(attrs, name) for name in attrs._v_attrnames}
        # We need to filter out the attributes that are not relevant for the calibration matrix.
        return {key: val for key, val in header_dict.items()
                if not key.isupper() and not key.startswith("PYTABLES_")}

    def _save_other_data(self, h5file: tables.File) -> None:
        """Save any other data that is specific to the derived class in the HDF5 file.
        """

    def to_hdf5(self, file_path: str) -> str:
        """Save the calibration matrix to an HDF5 file at the given path.

        Arguments
        ---------
        file_path : str
            The path of the file on the disk.
        """
        # pylint: disable=protected-access
        with tables.File(file_path, "w") as h5file:
            root = h5file.root
            # Save the matrix and the hits matrices as arrays in the HDF5 file.
            h5file.create_array(root, "matrix", self.matrix)
            h5file.create_array(root, "hits", self.hits)
            # Save any other data that is specific to the derived class.
            self._save_other_data(h5file)
            # Update the header with the relevant information.
            attrs = root._v_attrs
            attrs.num_rows = self._shape[0]
            attrs.num_cols = self._shape[1]
            attrs.default = self.default
            self._update_header(attrs)
        return file_path

    @classmethod
    def from_hdf5(cls, file_path: str) -> "CalibrationMatrixBase":
        """Create an instance of the calibration matrix from an HDF5 file at the given path.

        Arguments
        ---------
        file_path : str
            The path of the file on the disk.
        """
        # pylint: disable=protected-access
        with tables.File(file_path, "r") as h5file:
            # Load the attributes from the header.
            attrs = cls._load_header_dict(h5file.root._v_attrs)
            # Instantiate the object with the attributes loaded from the header.
            obj = cls(**attrs)
            # Loop over the nodes in the HDF5 file and set the corresponding attributes with the
            # data.
            for node in h5file.iter_nodes(h5file.root):
                node_name = node._v_name
                data = node[:]
                target_attr = f"_{node_name}"
                setattr(obj, target_attr, data)
        return obj


class CalibrationMatrixGain(CalibrationMatrixBase):

    """Gain calibration matrix for the detector.

    This class implements the logic to update the calibration matrix with signal events to
    determine the gain of each pixel. The way the gain is calculated from the data depends on the
    analysis method used. If the method is "single", only 1-pixel events are used, and the gain
    is estimated as the mean of the gain values for each pixel. If the method is "lsm", all the
    events are used, and a least squares fit is performed to determine the gain of each pixel.

    Arguments
    ---------
    num_cols : int
        The number of columns in the readout chip.
    num_rows : int
        The number of rows in the readout chip.
    energy : float
        The energy of the photons in the events used for the gain calibration, in eV.
    default : float, optional
        The default value to set for pixels in the calibration matrix. If None, the default value
        is estimated from the data.
    """

    def __init__(self, num_cols: int, num_rows: int, energy: float = None,
                 default: Optional[float] = None) -> None:
        """Class constructor.
        """
        super().__init__(num_cols, num_rows, default)
        self._energy = energy
        # Create the arrays to store the data for the least squares fit.
        self._event_count = 0
        self._pha = []
        self._coords = []
        self._event_rows = []
        self._fit = False

    @property
    def matrix(self) -> np.ndarray:
        """Return the calibration matrix.

        If the analysis method is "lsm", the calibration matrix is calculated from the data by
        performing a least squares fit to determine the gain of each pixel. Otherwise, the
        calibration matrix is calculated as the mean of the gain values for each pixel.
        """
        # Run the least squares fit, which updates the value of the calibration matrix.
        if self._event_count != 0 and not self._fit:
            self._lsm_fit()
            self._fit = True
        # Call the base class method to update the calibration matrix with the default value
        # for pixels with no events, and return the matrix.
        return super().matrix

    @matrix.setter
    def matrix(self, new_value: np.ndarray) -> None:
        if new_value.shape != self._shape:
            raise ValueError(f"Input matrix has shape {new_value.shape}, but expected shape is "
                             f"{self._shape}.")
        self._matrix = new_value
        # Setting the hits to one to avoid that the default value is estimated from the data.
        if not np.any(self._hits > 0):
            self._hits = np.ones(self._shape, dtype=int)

    def _update_header(self, attrs: AttributeSet) -> None:
        """Overloaded method.
        """
        attrs.energy = self._energy

    def _lsm_fit(self) -> None:
        """Perform the least squares fit to determine the gain of each pixel.
        """
        # Create the sparse matrix for the least squares fit. This object allows to store
        # and use efficiently the large and sparse matrix that we need for the fit.
        shape = (self._event_count, self._shape[0] * self._shape[1])
        a = csr_matrix((self._pha, (self._event_rows, self._coords)), shape=shape)
        # Create the vector of the expected number of electrons.
        b = np.full(self._event_count, self._energy / DEFAULT_IONIZATION_POTENTIAL)
        # Perform the fit
        results = lsmr(a, b)
        # Get the best-fit weight vector and reshape it to the shape of the calibration matrix.
        weight = results[0].reshape(self._shape)
        signal_power = np.array(a.multiply(a).sum(axis=0)).reshape(self._shape)
        # Calculate the number of degrees of freedom and the mean squared error.
        dof = self._event_count - (self._shape[0] * self._shape[1])
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
        # Set the gain value for the pixels that pass the quality cut.
        self._matrix[mask] = 1 / weight[mask]
        # Set the hits to zero for the pixels that don't pass the quality cut.
        self._hits[~mask] = 0

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
            self._hits[row, col] += 1
        # Update the event count
        self._event_count += 1


class CalibrationMatrixNoise(CalibrationMatrixBase):

    """Noise calibration matrix for the detector.

    This class implements the logic to update the calibration matrix with noise events to
    determine the noise characteristics.

    Arguments
    ---------
    num_cols : int
        The number of columns in the readout chip.
    num_rows : int
        The number of rows in the readout chip.
    default : float, optional
        The default value to set for pixels in the calibration matrix. If None, the default value
        is estimated as the mean of the noise distribution for each pixel.
    """

    def __init__(self, num_cols: int, num_rows: int, default: Optional[float] = None) -> None:
        """Class constructor.
        """
        super().__init__(num_cols, num_rows, default)
        # Create the array to store the histogram of the noise values. These data are useful
        # to estimate the width of the noise distribution.
        self._histogram = np.zeros(200, dtype=int)

    @property
    def histogram(self) -> np.ndarray:
        """Return the histogram of the noise values.
        """
        return self._histogram

    def enc(self) -> float:
        edges = np.arange(-0.5, len(self._histogram) + 0.5, 1)
        hist = Histogram1d(edges)
        hist.set_content(self._histogram)
        model = Gaussian()
        try:
            model.fit_iterative(hist)
        except RuntimeError:
            return hist.binned_statistics()[0]
        return model.sigma.value

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
        """Overloaded method.
        """
        # If the event is a bad event, don't update the calibration matrix and return it as it is.
        if self._bad_event(event):
            return
        # Otherwise, remove the pixels with signal from the event and update the matrices.
        noise_pha = self._remove_signal(event)
        row_slice, col_slice = event.roi.readout_slice()
        self._sum[row_slice, col_slice] += noise_pha
        self._hits[row_slice, col_slice][noise_pha > 0] += 1
        # Update the noise histogram
        counts = np.bincount(noise_pha[noise_pha > 0], minlength=len(self._histogram))
        self._histogram += counts[:len(self._histogram)]

    def _save_other_data(self, h5file: tables.File) -> None:
        """Save the noise counts histogram in the HDF5 file.
        """
        h5file.create_array(h5file.root, "histogram", self.histogram)


class ChargeFractionMatrices:

    """Charge fraction calibration matrices for the detector.

    This class implements the logic to create a set of calibration matrices to determine the
    fraction of charge collected by each pixel as a function of the incident position of the
    photon on the central pixel of the cluster. The calibration matrices are calculated by creating
    a grid of bins and calculating the average value of the fraction of charge collected by each
    pixel for each bin over all the events that fall in that bin.

    Arguments
    ---------
    nbins : int
        The number of bins in the x and y axes of the grid. The grid is a square grid.
    readout : HexagonalReadoutBase
        The detector readout instance.
    """

    def __init__(self, nbins: int, readout: HexagonalReadoutBase) -> None:
        """Class constructor.
        """
        self.nbins = nbins
        # Initialize the arrays to store the calibration data and the bin edges.
        self._xbins = None
        self._ybins = None
        self._matrices = np.zeros((7, nbins, nbins))
        # Set the bin edges according to the pixel orientation.
        if readout:
            if readout.pointy_topped():
                self.xedges = np.linspace(-0.5, 0.5, nbins + 1)
                self.yedges = np.linspace(-1/np.sqrt(3), 1/np.sqrt(3), nbins + 1)
            if readout.flat_topped():
                self.xedges = np.linspace(-1/np.sqrt(3), 1/np.sqrt(3), nbins + 1)
                self.yedges = np.linspace(-0.5, 0.5, nbins + 1)
            # Calculate the bin centers from the edges.
            self._x_bins = (self.xedges[:-1] + self.xedges[1:]) / 2
            self._y_bins = (self.yedges[:-1] + self.yedges[1:]) / 2

    def upload_data(self, x: np.ndarray, y: np.ndarray, fraction: np.ndarray) -> None:
        """Update the calibration matrix with the data from the events. The data are uploaded by
        calculating the average value of the charge fraction for each bin in the x and y axes, and
        storing the average values in the corresponding bins of the calibration matrix.

        Arguments
        ---------
        x : np.ndarray
            The x coordinates of the events, in units of pixel pitch.
        y : np.ndarray
            The y coordinates of the events, in units of pixel pitch.
        fraction : np.ndarray
            The charge fraction values array of the events.
        """
        bin_count, _, _ = np.histogram2d(x, y, bins=[self.xedges, self.yedges])
        for i in range(7):
            bin_sum, _, _ = np.histogram2d(x, y, bins=[self.xedges, self.yedges],
                                           weights=fraction[:, i])
            with np.errstate(divide='ignore', invalid='ignore'):
                average = bin_sum / bin_count
                average[np.isnan(average)] = 0
            self._matrices[i, :, :] = average

    @property
    def matrices(self) -> np.ndarray:
        """Set of calibration matrices with the fraction of charge collected by each pixel for each
        position of the grid.
        """
        return self._matrices

    @property
    def x_bins(self) -> np.ndarray:
        """Bin centers in the x axis.
        """
        return self._x_bins

    @property
    def y_bins(self) -> np.ndarray:
        """Bin centers in the y axis.
        """
        return self._y_bins

    def to_hdf5(self, file_path: str) -> str:
        """Save the calibration matrices to an HDF5 file at the given path. The file stores the
        calibration matrices and the bin centers in the x and y axes.

        Arguments
        ---------
        file_path : str
            The path of the file on the disk.
        
        Returns
        -------
        file_path : str
            The path of the file on the disk.
        """
        with tables.File(file_path, "w") as h5file:
            root = h5file.root
            h5file.create_array(root, "matrices", self.matrices)
            h5file.create_array(root, "x_bins", self.x_bins)
            h5file.create_array(root, "y_bins", self.y_bins)
        return file_path

    @classmethod
    def from_hdf5(cls, file_path: str) -> "ChargeFractionMatrices":
        """Create an instance of ChargeFractionMatrices from an HDF5 file at the given path. The
        instance contains the calibration matrices and the bin centers in the x and y axes.

        Arguments
        ---------
        file_path : str
            The path of the file on the disk.
        
        Returns
        -------
        obj : ChargeFractionMatrices
            An instance of ChargeFractionMatrices initialized with the data from the HDF5 file.
        """
        with tables.File(file_path, "r") as h5file:
            matrices = h5file.root.matrices[:]
            x_bins = h5file.root.x_bins[:]
            y_bins = h5file.root.y_bins[:]
        # Instantiate the object with the data loaded from the HDF5 file.
        obj = cls(0, None)
        # Set the attributes with the data loaded from the HDF5 file.
        obj._matrices = matrices
        obj._x_bins = x_bins
        obj._y_bins = y_bins
        return obj


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
