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


import numpy as np
import tables
from aptapy.hist import Histogram2d
from aptapy.models import Probit
from aptapy.plotting import last_line_color, plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr
from tables.attributeset import AttributeSet
from tqdm import tqdm

from hexsample.hexagon import HexagonalGrid

from .clustering import Cluster, ClusteringNN
from .digi import DigiEventRectangular
from .fileio import DigiInputFileBase


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
    default : float | None
        The default value to set for pixels in the calibration matrix.
    """

    def __init__(self, num_cols: int, num_rows: int, default: float | None = None,
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
        # Otherwise, the default value is estimated from the data, by calculating the mean of the
        # calibration matrix.
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
            attrs.shape = self._shape
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
            # Load the matrices from the HDF5 file.
            matrix = h5file.root.matrix[:]
            hits = h5file.root.hits[:]
            # Load the attributes from the header and return the calibration matrix object.
            attrs = h5file.root._v_attrs
            shape = attrs.shape
            num_cols, num_rows = shape[1], shape[0]
            if hasattr(attrs, "energy"):
                obj = cls(num_cols, num_rows, energy=attrs.energy, default=attrs.default)
            else:
                obj = cls(num_cols, num_rows, default=attrs.default)
                obj.histogram = h5file.root.histogram[:]
            obj._matrix = matrix
            obj._hits = hits
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
    default : float | None
        The default value to set for pixels in the calibration matrix. If None, the default value
        is estimated from the data.
    method : str
        The method to use for the gain calibration. Choices are "single", which uses only 1-pixel,
        and "lsm", which uses all the events and performs a least squares fit. Default is "lsm".
    """

    def __init__(self, num_cols: int, num_rows: int, energy: float = None,
                 default: float | None = None, method: str = None) -> None:
        """Class constructor.
        """
        super().__init__(num_cols, num_rows, default)
        self._energy = energy
        self._method = method
        # Create the arrays to store the data for the least squares fit.
        self._event_count = 0
        self._pha = []
        self._coords = []
        self._event_rows = []

    @property
    def matrix(self) -> np.ndarray:
        """Return the calibration matrix.

        If the analysis method is "lsm", the calibration matrix is calculated from the data by
        performing a least squares fit to determine the gain of each pixel. Otherwise, the
        calibration matrix is calculated as the mean of the gain values for each pixel.
        """
        if self._method == "lsm":
            # Run the least squares fit, which updates the value of the calibration matrix.
            self._lsm_fit()
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
        b = np.full(self._event_count, self._energy / 3.6)
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

    def analyze_cluster(self, cluster: Cluster, grid: HexagonalGrid):
        """Analyze the event cluster to update the calibration matrix.
        """
        # If the analysis method is single, only 1-pixel events are used for the gain calibration.
        if self._method == "single":
            if cluster.size() == 1:
                # Get the coordinate of the only pixel of the cluster
                col, row = grid.world_to_pixel(cluster.x[0], cluster.y[0])
                # The gain is estimated as the ADC counts of the pixel divided by the expected
                # number of electrons for the given energy.
                self._sum[row, col] += cluster.pha[0] / (self._energy / 3.6)
                self._hits[row, col] += 1
        # If the analysis method is lsm, all the events (which are 1, 2 and 3-pixel events) are
        # used for the calibration.
        elif self._method == "lsm":
            # Get the coordinates of the cluster pixels
            cols, rows = grid.world_to_pixel(cluster.x, cluster.y)
            # Update the arrays for the least squares fit.
            self._pha.extend(cluster.pha)
            for col, row in zip(cols, rows):
                # Calculate the index of the pixel in the flattened array
                i = row * grid.num_cols + col
                self._coords.append(i)
                self._event_rows.append(self._event_count)
                # Update the matrix with the number of events for each pixel
                self._hits[row, col] += 1
            # Update the event count
            self._event_count += 1
        else:
            raise ValueError(f"Unknown method {self._method} for gain calibration.")


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
    default : float | None
        The default value to set for pixels in the calibration matrix. If None, the default value
        is estimated as the mean of the noise distribution for each pixel.
    """

    def __init__(self, num_cols: int, num_rows: int, default: float | None = None) -> None:
        """Class constructor.
        """
        super().__init__(num_cols, num_rows, default)
        # Create the array to store the histogram of the noise values. These data are useful
        # to estimate the width of the noise distribution.
        self.histogram = np.zeros(50, dtype=int)

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

    def analyze_event(self, event: DigiEventRectangular):
        """Overloaded method.
        """
        # If the event is a bad event, don't update the calibration matrix and return it as it is.
        if self._bad_event(event):
            return self
        # Otherwise, remove the pixels with signal from the event and update the matrices.
        noise_pha = self._remove_signal(event)
        row_slice, col_slice = event.roi.readout_slice()
        self._sum[row_slice, col_slice] += noise_pha
        self._hits[row_slice, col_slice][noise_pha > 0] += 1
        # Update the noise histogram
        counts = np.bincount(noise_pha[noise_pha > 0], minlength=len(self.histogram))
        self.histogram += counts[:len(self.histogram)]
        return self

    def _save_other_data(self, h5file: tables.File) -> None:
        """Save the noise counts histogram in the HDF5 file.
        """
        h5file.create_array(h5file.root, "histogram", self.histogram)


def profile(xdata: np.ndarray, ydata: np.ndarray, xbins: int, ybins: int
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    # Be sure that the input data are float arrays
    xdata = xdata.astype(float)
    ydata = ydata.astype(float)
    # Create the 2D histogram to compute the profile
    xedges = np.linspace(xdata.min(), xdata.max(), xbins + 1)
    yedges = np.linspace(ydata.min(), ydata.max(), ybins + 1)
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


def distance(pos: np.ndarray, projection_axis: np.ndarray | None = None) -> np.ndarray:
    """Calculate the distance of the photon from the center of the most charged pixel. If
    specified, project the distance on the given projection axis, given as a unit vector.

    Arguments
    ---------
    pos : np.ndarray
        The position of the photon with respect to the most charged pixel, in units of pitch.
    projection_axis : np.ndarray | None
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


def calibration_data(input_file: DigiInputFileBase, clustering: ClusteringNN, pitch: float
                     ) -> tuple[np.ndarray, ...]:
    """Open the simulated input file and extract the data needed for the calibration of the eta
    function. The data are extracted only for 2-pixel and 3-pixel clusters. The resuling arrays
    need to be masked to select the desired cluster size before the calibration.

    Arguments
    ---------
    input_file : DigiInputFileBase
        The input file to be analyzed.
    clustering : ClusteringNN
        The clustering algorithm to be used to reconstruct the clusters.
    pitch : float
        The pixel pitch of the detector.
    
    Returns
    -------
    size : np.ndarray
        Array containing the size of the clusters for each event.
    photon_pos : np.ndarray
        Array containing the position of the photon with respect to the most charged pixel, in
        units of pitch, for each event.
    versors : np.ndarray
        Array containing the versors of the cluster for each event.
    eta : np.ndarray
        Array containing the eta values for each event.
    """
    # Create the lists to store the data.
    size_list, photon_pos_list, versors_list, eta_list = [], [], [], []
    # Loop over the events and calculate the interesting quantities.
    for i, event in tqdm(enumerate(input_file)):
        cluster = clustering.run(event)
        # Analyze only 2-pixel and 3-pixel events.
        if cluster.size() == 2 or cluster.size() == 3:
            mc_event = input_file.mc_event(i)
            size_list.append(cluster.size())
            # Calculate the photon position with respect to the most charged pixel
            ph_pos = np.array([mc_event.absx - cluster.x[0],
                               mc_event.absy - cluster.y[0]]) / pitch
            photon_pos_list.append(ph_pos)
            eta_list.append(cluster.calculate_eta())
            versors_list.append(cluster.versors())
    # Convert the lists to numpy arrays
    size = np.asarray(size_list, dtype=int)
    photon_pos = np.asarray(photon_pos_list, dtype=float)
    versors = np.asarray(versors_list, dtype=float)
    eta = np.asarray(eta_list, dtype=object)
    return size, photon_pos, versors, eta


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


def calibrate_dr_3pix(eta: np.ndarray, dr: np.ndarray, nbins: int, **kwargs) -> tuple[float, float]:
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
    eta_sum = np.sum(eta, axis=1)
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
