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

from abc import ABC, abstractmethod

import numpy as np
import tables
from tables.attributeset import AttributeSet
from aptapy.hist import Histogram2d
from aptapy.models import Probit
from aptapy.plotting import last_line_color, plt
from tqdm import tqdm

from .clustering import ClusteringNN
from .digi import DigiEventRectangular
from .fileio import DigiInputFileBase



class CalibrationMatrixBase(ABC):

    """Abstract base class for calibration analysis.

    This includes the common structure of the detector calibration matrices, which are used to
    calibrate each pixel of the detector independently, and the facilities to save and load
    calibration data from HDF5 files.
    The derived classes need to implement the logic to update the calibration matrix with info
    from the digitized events. They also need to implement the logic to determine how to estimate
    the default value of the matrix for pixels with no events, if no default value is provided.

    Arguments
    ---------
    num_cols : int
        The number of columns in the readout chip.
    num_rows : int
        The number of rows in the readout chip.
    default : float | None
        The default value to set for pixels in the calibration matrix.
    """

    def __init__(self, num_cols: int, num_rows: int, default: float | None = None) -> None:
        """Class constructor.
        """
        self._shape = (num_rows, num_cols)
        # Create the arrays to store the calibration data and the number of events for each pixel.
        self._sum = np.zeros(self._shape)
        self.num_events = np.zeros(self._shape, dtype=int)
        self._default = default

    @property
    @abstractmethod
    def default(self) -> float:
        """Calculate the default value of the calibration matrix for pixels with no events.
        The logic to estimate the default value is implemented in the derived classes, as it
        depends on the particular quantity to calibrate.
        """
        pass

    @property
    def value(self) -> np.ndarray:
        """Calculate the mean value of the calibration matrix for each pixel, by dividing the sum
        of the values by the number of events for each pixel.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(self.num_events > 0, self._sum / self.num_events, self.default)

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

    @abstractmethod
    def _update_header(self, attrs: AttributeSet):
        """Update the header of the HDF5 file with the relevant information for the calibration
        matrix.
        """
        pass

    def to_hdf5(self, file_path: str) -> str:
        """Save the calibration matrix to an HDF5 file at the given path.

        Arguments
        ---------
        file_path : str
            The path of the file on the disk.
        """
        with tables.File(file_path, "w") as h5file:
            root = h5file.root
            # Save the value and the number of events matrices as arrays in the HDF5 file.
            h5file.create_array(root, "value", self.value)
            h5file.create_array(root, "num_events", self.num_events)
            # Update the header with the relevant information.
            self._update_header(root._v_attrs)
        return file_path

    @classmethod
    def from_hdf5(cls, file_path: str) -> "CalibrationMatrixBase":
        """Create an instance of the calibration matrix from an HDF5 file at the given path.

        Arguments
        ---------
        file_path : str
            The path of the file on the disk.
        """
        with tables.File(file_path, "r") as h5file:
            # Load the matrices from the HDF5 file.
            value = h5file.root.value[:]
            num_events = h5file.root.num_events[:]
            # Load the attributes from the header and return the calibration matrix object.
            attrs = h5file.root._v_attrs
            if hasattr(attrs, "energy") and hasattr(attrs, "zero_sup_threshold"):
                obj = cls(value.shape[1], value.shape[0],
                          energy=attrs.energy,
                          zero_sup_threshold=attrs.zero_sup_threshold)
            else:
                obj = cls(value.shape[1], value.shape[0])
            # With this logic we can update the instance with new events, as it has all the info
            # to reconstruct the calibration matrix (_sum and num_events).
            obj._sum = value * num_events
            obj.num_events = num_events
        return obj

    @abstractmethod
    def __iadd__(self, event: DigiEventRectangular):
        """Update the calibration matrix with the information from a digitized event. The logic to
        update the matrix is implemented in the derived classes, as it depends on the particular
        calibration analysis.

        Arguments
        ---------
        event : DigiEventRectangular
            The event to be analyzed.
        """
        pass


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

    @property
    def default(self) -> float:
        """Calculate the default value of the noise level for pixels with no events. If a default
        value is provided in the constructor, set that value as the default. Otherwise, its value
        is estimated using the relation between the mean and the sigma of half of a normal
        distribution with mean 0 and sigma equal to the noise level.
        """
        # If the default value is provided, return it.
        if self._default is not None:
            return self._default
        # If the default value is not provided, but the calibration matrix has no events, return 0.
        if not np.any(self.num_events > 0):
            return 0.
        # Otherwise, estimate it from the data
        with np.errstate(divide='ignore', invalid='ignore'):
            tmp_value = self._sum[self.num_events > 0] / self.num_events[self.num_events > 0]
            mean_noise = np.mean(tmp_value)
            return mean_noise / (2 / np.pi) ** 0.5

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
    
    def _update_header(self, attrs: AttributeSet) -> None:
        """Update the header of the HDF5 file with the relevant information for the noise
        calibration matrix.

        Arguments
        ---------
        attrs : AttributeSet
            The AttributeSet object to be updated with the relevant information for the noise
        """
        attrs.default = self.default

    def __iadd__(self, event: DigiEventRectangular):
        """Overloaded method.
        """
        # If the event is a bad event, don't update the calibration matrix and return it as it is.
        if self._bad_event(event):
            return self
        # Otherwise, remove the pixels with signal from the event and update the matrices.
        noise_pha = self._remove_signal(event)
        row_slice, col_slice = event.roi.readout_slice()
        self._sum[row_slice, col_slice] += noise_pha
        self.num_events[row_slice, col_slice][noise_pha > 0] += 1
        return self


class CalibrationMatrixGain(CalibrationMatrixBase):
    
    """Gain calibration matrix for the detector.

    This class implements the logic to update the calibration matrix with signal events to
    determine the gain of each pixel. The gain matrix is calculated using only 1-pixel events and
    the energy of the photons in the events.

    Arguments
    ---------
    num_cols : int
        The number of columns in the readout chip.
    num_rows : int
        The number of rows in the readout chip.
    default : float | None
        The default value to set for pixels in the calibration matrix. If None, the default value
        is estimated from the data.
    energy : float
        The energy of the photons in the events used for the gain calibration, in eV.
    zero_sup_threshold : float
        The zero suppression threshold used in the clustering of the events, in ADC counts.
    """

    def __init__(self, num_cols: int, num_rows: int, default: float | None, energy: float,
                 zero_sup_threshold: float) -> None:
        """Class constructor.
        """
        super().__init__(num_cols, num_rows, default)
        self._energy = energy
        self._zero_sup_threshold = zero_sup_threshold

    @property
    def default(self) -> float:
        """Calculate the default value of the gain for pixels with no events. If a default value is
        provided in the constructor, set that value as the default. Otherwise, its value is
        estimated from the data, by calculating the mean of the gain distribution for pixels
        """
        if self._default is not None:
            return self._default
        # If the default value is not provided, but the calibration matrix has no events, return 0.
        if not np.any(self.num_events > 0):
            return 0.
        # Otherwise, estimate it from the data, by calculating the mean of the gain distribution.
        with np.errstate(divide='ignore', invalid='ignore'):
            tmp_value = self._sum[self.num_events > 0] / self.num_events[self.num_events > 0]
            return np.mean(tmp_value)

    def _cluster_size(self, event) -> int:
        """Estimate the cluster size counting the number of pixels above the zero suppression
        threshold in the 3x3 region around the highest pixel in the event. This size is not
        very accurate, because this region contains 9 pixels, while the maximum cluster size
        is 7 pixels. However, this is a fast way to estimate the cluster size without the need
        to run the clustering algorithm, which slows down the analysis significantly.

        This way we may loose some statistics. Since we are interested in calibrating the
        gain with 1-pixel events only, we do not care about overestimating the cluster size.

        Arguments
        ---------
        event : DigiEventRectangular
            The event to be analyzed.
        """
        pha = event.pha.copy()
        # Remove the pixels below the zero suppression threshold.
        pha[pha < self._zero_sup_threshold] = 0
        # Get the highest pixel coordinates in the event.
        seed_col, seed_row = event.highest_pixel(absolute=False)
        size = np.count_nonzero(pha[seed_row - 1: seed_row + 2, seed_col - 1: seed_col + 2])
        return size

    def _update_header(self, attrs: AttributeSet) -> None:
        """Update the header of the HDF5 file with the relevant information for the gain
        calibration matrix.

        Arguments
        ---------
        attrs : AttributeSet
            The AttributeSet object to be updated with the relevant information for the gain
        """
        attrs.default = self.default
        attrs.energy = self._energy
        attrs.zero_sup_threshold = self._zero_sup_threshold

    def __iadd__(self, event: DigiEventRectangular):
        """Overloaded method.
        """
        # If the event is a bad event, or if the cluster size is not 1, don't update the
        # calibration matrix and return it as it is.
        if self._cluster_size(event) != 1 or self._bad_event(event):
            return self
        # Otherwise, update the calibration matrix.
        col, row = event.highest_pixel(absolute=True)
        seed_col, seed_row = event.highest_pixel(absolute=False)
        # The gain is estimated as the ADC counts of the pixel divided by the expected number of
        # electrons for the given energy.
        self._sum[row, col] += event.pha[seed_row, seed_col] / (self._energy / 3.6)
        self.num_events[row, col] += 1
        return self


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
