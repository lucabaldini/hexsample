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

"""Position reconstruction facilities for calibration and evaluation.
"""

import pathlib
from enum import Enum

import h5py
import numpy as np

from .calibration import CalibrationType
from .clustering import Cluster
from .hexagon import HexagonalGrid
from .mc import MonteCarloEvent
from .stats import RunningStats


class MLECalibrationMetadata(str, Enum):

    """Enum to store the metadata keys for the MLE calibration data.
    """

    CALIBRATION_TYPE = "calibration_type"
    DIFFUSION_SIGMA = "diffusion_sigma"
    FILE_NAME = "file_name"
    LAYOUT = "layout"
    PITCH = "pitch"
    THICKNESS = "thickness"
    VERSION = "version"


class MLECalibrationData:

    """Class to store and use calibration data for Maximum Likelihood Estimation
    (MLE) position reconstruction.

    This class stores a set of seven matrices, each containing the average fraction
    of charge collected by a pixel in a cluster of 7 pixels as a function of the incident
    position of the photon on the central pixel of the cluster.
    """

    VALUES = "values"
    X_BINS = "x_bins"
    Y_BINS = "y_bins"

    def __init__(self, x_bins: np.ndarray, y_bins: np.ndarray) -> None:
        self._x_bins = x_bins
        self._y_bins = y_bins
        # Create the tensor to store the calibration data.
        self._values = np.zeros((7, len(x_bins), len(y_bins)))
        # Some useful information for the metadata.
        self._metadata = {
            MLECalibrationMetadata.CALIBRATION_TYPE: CalibrationType.MLE.value
        }

    @property
    def values(self) -> np.ndarray:
        """Calibration data tensor of dimensions (7, x_bins, y_bins) containing
        the average fraction of charge collected by each pixel.
        """
        return self._values

    @values.setter
    def values(self, new_values: np.ndarray) -> None:
        """Set the calibration data tensor with the new values.
        """
        if new_values.shape != self._values.shape:
            raise ValueError(f"Invalid shape for the new values: {new_values.shape}. "
                             f"Expected shape: {self._values.shape}.")
        self._values = new_values

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
    def metadata(self) -> dict:
        """Metadata dictionary containing useful information about the calibration data.
        """
        return self._metadata

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

    def to_hdf5(self, file_path: str) -> str:
        """Save the calibration data to an HDF5 file at the given path.


        Arguments
        ---------
        file_path : str
            The path of the file on the disk.
        """
        # Define the compression parameters for the HDF5 datasets.
        compression_pars = dict(
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )
        # Save the calibration data and the bin centers in the x and y axes.
        with h5py.File(file_path, "w") as h5file:
            h5file.create_dataset(self.VALUES, data=self._values, **compression_pars)
            h5file.create_dataset(self.X_BINS, data=self._x_bins, **compression_pars)
            h5file.create_dataset(self.Y_BINS, data=self._y_bins, **compression_pars)
            self.update_metadata(MLECalibrationMetadata.FILE_NAME, pathlib.Path(file_path).stem)
            # Save the metadata in the file header.
            for key, val in self.metadata.items():
                h5file.attrs[key] = val
        return file_path

    @classmethod
    def from_hdf5(cls, file_path: str) -> "MLECalibrationData":
        """Load the calibration data from an HDF5 file at the given path.

        Arguments
        ---------
        file_path : str
            The path of the file on the disk.
        """
        # Check if the file path is valid and the file exists.
        if file_path is None:
            raise ValueError("No file path provided for the MLE calibration data.")
        if not pathlib.Path(file_path).is_file():
            raise ValueError(f"File not found: {file_path}")
        with h5py.File(file_path, "r") as h5file:
            # Load the attributes from the header.
            attrs = dict(h5file.attrs)
            # Instantiate the object and set the attributes.
            x_bins = h5file[cls.X_BINS][:]
            y_bins = h5file[cls.Y_BINS][:]
            obj = cls(x_bins=x_bins, y_bins=y_bins)
            obj._values = h5file[cls.VALUES][:]
            # Set the metadata.
            for key, val in attrs.items():
                obj._metadata[key] = val
        return obj

    def __str__(self):
        """Return a string representation of the MLECalibrationData object.
        """
        if MLECalibrationMetadata.FILE_NAME in self._metadata:
            return self._metadata[MLECalibrationMetadata.FILE_NAME]
        return f"MLECalibrationData(shape={self.values.shape})"


class CalibrateMLE:

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
    grid: HexagonalGrid
        The hexagonal grid with the same geometry of the detector.
    """

    PIXEL_SIZE = dict(
        flat_topped=(2 / np.sqrt(3), 1),
        pointy_topped=(1, 2 / np.sqrt(3))
    )

    def __init__(self, bin_size: float, grid: HexagonalGrid) -> None:
        """Class constructor.
        """
        if bin_size <= 0 or bin_size > 1:
            raise ValueError(f"Invalid bin size: {bin_size}. Bin size must be between (0, 1].")
        self.bin_size = bin_size
        self.grid = grid
        # Calculate the bin edges according to the pixel orientation.
        if grid.flat_topped():
            x_size, y_size = self.PIXEL_SIZE["flat_topped"]
        else:
            x_size, y_size = self.PIXEL_SIZE["pointy_topped"]
        x_nbins = int(x_size / bin_size)
        y_nbins = int(y_size / bin_size)
        xedges = np.linspace(-x_size / 2, x_size / 2, x_nbins + 1)
        yedges = np.linspace(-y_size / 2, y_size / 2, y_nbins + 1)
        # Store bin edges for digitize and calculate the bin centers from the edges.
        self._x_edges = xedges
        self._y_edges = yedges
        self.x_bins = (xedges[:-1] + xedges[1:]) / 2
        self.y_bins = (yedges[:-1] + yedges[1:]) / 2
        # Create the MLECalibrationData object to store the calibration data.
        self.cal_data = MLECalibrationData(x_bins=self.x_bins, y_bins=self.y_bins)
        self._stats = RunningStats(shape=self.cal_data.values.shape)

    def analyze_cluster(self, cluster: Cluster, mc_event: MonteCarloEvent) -> None:
        """Analyze the event cluster and update the calibration data.

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

    def fit(self) -> MLECalibrationData:
        """Calculate the average charge fractions for each bin and store them in the
        MLECalibrationData object.

        Returns
        -------
        cal_data : MLECalibrationData
            The MLECalibrationData object containing the calibration data.
        """
        self.cal_data.values = self._stats.mean()
        return self.cal_data
