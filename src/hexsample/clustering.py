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

"""Clustering facilities.
"""

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from .digi import DigiEventCircular, DigiEventRectangular
from .position import eta_2pix, eta_3pix, mle
from .readout import HexagonalReadoutBase

# This line is necessary to avoid circular imports errors, allowing to import the class only
# when type checking is performed
if TYPE_CHECKING:
    from .calibration import CalibrationMatrix, PositionCalibrationData


@dataclass
class Cluster:

    """Small container class describing a cluster.
    """

    # pylint: disable = invalid-name

    x: np.ndarray
    y: np.ndarray
    col: np.ndarray
    row: np.ndarray
    pha: np.ndarray
    adc_to_ev: float
    pos_recon_algorithm: str
    recon_pars: Optional[dict] = None

    def __post_init__(self) -> None:
        """Small cross check on the dimensions of the arrays passed in the constructor.
        """
        if not self.x.shape == self.y.shape == self.pha.shape:
            raise RuntimeError(f"Inconsistent arrays: x = {self.x}, y = {self.y}, pha = {self.pha}")

    def size(self) -> int:
        """Return the size of the cluster.
        """
        return self.x.size

    def pulse_height(self) -> float:
        """Return the total pulse height of the cluster.
        """
        return self.pha.sum()

    def energy(self) -> float:
        """Return the energy of the cluster in eV.
        """
        return self.pulse_height() * self.adc_to_ev

    def centroid(self) -> Tuple[float, float]:
        """Return the cluster centroid.
        """
        return np.average(self.x, weights=self.pha), np.average(self.y, weights=self.pha)

    def eta(self, position_cal: "PositionCalibrationData", pitch: float) -> Tuple[float, float]:
        """Return the cluster reconstructed position using the eta function
        calibrated for 2 and 3 pixel clusters.
        
        .. note::
            If cluster size is not 2 or 3, the position is reconstructed using the
            centroid algorithm.

        Arguments
        ---------
        position_cal : PositionCalibrationData
            The position calibration data containing the parameters for the eta
            reconstruction algorithm.
        
        pitch : float
            The pixel pitch.
        """
        # Calculate the size of the cluster, to choose the correct reconstruction
        # method.
        size = self.size()
        # If size is 2 or 3, we use the corresponding eta reconstruction method...
        if size == 2:
            dx, dy = eta_2pix(self.pha, self.x, self.y, position_cal.two_pix_rad_sigma)
        elif size == 3:
            args = (position_cal.three_pix_rad_offset, position_cal.three_pix_rad_sigma,
                    position_cal.three_pix_theta_sigma)
            dx, dy = eta_3pix(self.pha, self.x, self.y, *args)
        # ... otherwise use the centroid algorithm.
        else:
            return self.centroid()
        # Calculate the absolute position of the photon.
        return self.x[0] + dx * pitch, self.y[0] + dy * pitch

    def mle(self,
            position_cal: "PositionCalibrationData",
            noise_matrix: "CalibrationMatrix",
            equalization_matrix: "CalibrationMatrix",
            pitch: float
            ) -> Tuple[float, float]:
        """Return the cluster reconstructed position using the maximum likelihood estimator.

        Arguments
        ---------
        mle_data : MLECalibrationData
            The MLE calibration data containing the precomputed charge fraction
            matrices and other relevant information.

        noise_matrix : CalibrationMatrix
            The noise calibration matrix containing the equalized noise standard
            deviation for each pixel.

        equalization_matrix : CalibrationMatrix
            The equalization calibration matrix containing the gain correction
            for each pixel.

        pitch : float
             The pixel pitch.
        """
        # Calculate the equalized noise for the pixels in the cluster.
        equal_noise = noise_matrix(self.col, self.row) / equalization_matrix(self.col, self.row)
        # Calculate the initial guess for the position of the photon, using the
        # centroid of the cluster.
        p0 = (self.centroid() - np.array([self.x[0], self.y[0]])) / pitch
        # Run the minimization.
        m = mle(self.pha, equal_noise, position_cal.values, position_cal.bin_size,
                position_cal.xlims, position_cal.ylims, p0=p0)
        # Calculate the absolute position of the photon from the fit results.
        return self.x[0] + m.values["x"] * pitch, self.y[0] + m.values["y"] * pitch

    def position(self) -> Tuple[float, float]:
        """Return the cluster reconstructed position using the position reconstruction
        algorithm specified in the constructor.
        """
        # Get the reconstruction algorithm callable from the class attributes.
        recon_algorithm = getattr(self, self.pos_recon_algorithm, None)
        if recon_algorithm is None:
            raise AttributeError(f"Invalid position reconstruction algorithm \
                                {self.pos_recon_algorithm}")
        # Get the arguments of the reconstruction algorithm method.
        args = inspect.signature(recon_algorithm).parameters.keys()
        # Create a dictionary with the arguments to be passed to the method.
        filtered_kwargs = {k: v for k, v in self.recon_pars.items() if k in args}
        return recon_algorithm(**filtered_kwargs)


@dataclass
class ClusteringBase:

    """Base class for the clustering.
    """

    readout: HexagonalReadoutBase
    zero_sup_threshold: float

    @staticmethod
    def zero_suppress(array: np.ndarray, threshold: np.ndarray) -> np.ndarray:
        """Zero suppress a generic array.
        """
        out = array.copy()
        out[out <= threshold] = 0
        return out

    def position_suppress(self, pha: np.ndarray, col: np.ndarray, row: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Suppress pixels in the cluster that do not satisfy the position requirements.

        If the cluster contains 2 or fewer pixels, no action is taken. For clusters with
        more than 2 pixels, the algorithm retains the two most charged pixels and
        only one additional neighbor (the one with the highest charge) of the second
        pixel, discarding all others.

        Arguments
        ---------
        pha : np.ndarray
            The array of pulse heights of the pixels in the cluster, ordered in decreasing order.

        col : np.ndarray
            The array of column indexes of the pixels in the cluster.

        row : np.ndarray
            The array of row indexes of the pixels in the cluster.

        Returns
        -------
        pha : np.ndarray
            The array of pulse heights of the pixels in the cluster after position suppression,
            ordered in decreasing order.

        col : np.ndarray
            The array of column indexes of the pixels in the cluster after position suppression,
            ordered in decreasing order of pulse height.

        row : np.ndarray
            The array of row indexes of the pixels in the cluster after position suppression,
            ordered in decreasing order of pulse height.
        """
        # If we have 2 or less pixels above threshold, we can't do anything, just remove the zeros.
        if np.count_nonzero(pha) <= 2 or pha.size <= 2:
            mask = pha > 0
            return pha[mask], col[mask], row[mask]
        # For events with more than 2 pixels above threshold, we keep only the highest neighbor of
        # the second pixel.
        pix = list(zip(col, row))
        # We find the neighbors of the second pixel.
        neighbors = set(self.readout.neighbors(col[1], row[1]))
        # We always keep the first two pixels, plus the two neighbors of the second pixel.
        mask = np.array([True, True] + [p in neighbors for p in pix[2:]], dtype=bool)
        mask = mask & (pha > 0)
        # Throw away the pixels that are not neighbors of the second pixel and that are zero.
        out_pha = pha[mask]
        out_col = col[mask]
        out_row = row[mask]
        # Sort the arrays in decreasing order of pulse height.
        idx = np.argsort(-out_pha)[:3]
        return out_pha[idx], out_col[idx], out_row[idx]

    def run(self, event: DigiEventRectangular) -> Cluster:
        """Workhorse method to be reimplemented by derived classes.
        """
        raise NotImplementedError


@dataclass
class ClusteringNN(ClusteringBase):

    """Neirest neighbor clustering.

    This is a very simple clustering strategy where we use the highest pixel in
    the event as a seed, loop over the six neighbors (after the zero suppression)
    and keep the N highest pixels.

    Arguments
    ---------
    num_neighbors : int
        The number of neighbors (between 0 and 6) to include in the cluster.
    pos_recon_algorithm : str
        The position reconstruction algorithm to use for the cluster position reconstruction.
        Possible values are "centroid" and "eta".
    recon_pars : dict, optional
        The parameters for the position reconstruction algorithm. This is not required if
        pos_recon_algorithm is "centroid".
    """

    num_neighbors: int
    pos_recon_algorithm: str
    recon_pars: Optional[dict] = None

    def run(self, event) -> Optional[Cluster]:
        """Overloaded method.

        .. warning::
           The loop ever the neighbors might likely be vectorized and streamlined
           for speed using proper numpy array for the offset indexes.
        """
        # Load the readout calibration matrices.
        noise = self.readout.enc
        pedestal = self.readout.pedestal
        gain = self.readout.gain
        # Load the adc_to_ev conversion factor from the readout metadata of the
        # equalization matrix. If the data is not present, or the wrong matrix type
        # is passed, this will raise a KeyError.
        adc_to_ev = gain.metadata["adc_to_ev"]
        if isinstance(event, DigiEventCircular):
            # If the readout is circular, we want to take all the neirest neighbors.
            # Trailing -1 is bc the central px is already considered.
            self.num_neighbors = 6 #HexagonalReadoutCircular.NUM_PIXELS - 1
            seed_coords = (event.column, event.row)
            if self.readout.is_at_border(*seed_coords):
                return None
            # Taking the NN logical coordinates ...
            neigh_coords = self.readout.neighbors(*seed_coords)
            col, row = np.vstack((seed_coords, neigh_coords)).T
            # ... transforming the coordinates in the corresponding ADC channel ...
            adc_channel_order = self.readout.adc_channel(col, row)
            # ... reordering the pha array for the correspondance (col[i], row[i]) with pha[i]
            # and applying pedestal and gain correction.
            pha = (event.pha[adc_channel_order] - pedestal(col, row)) / gain(col, row)
        elif isinstance(event, DigiEventRectangular):
            seed_coords = event.highest_pixel()
            if self.readout.is_at_border(*seed_coords):
                return None
            neigh_coords = self.readout.neighbors(*seed_coords)
            col, row = np.vstack((seed_coords, neigh_coords)).T
            pha = (event(col, row) - pedestal(col, row)) / gain(col, row)
        else:
            raise RuntimeError(f"Unsupported event type {type(event)} for clustering")
        # Zero suppressing the event (whatever the readout type)...
        threshold = self.zero_sup_threshold * (noise(col, row) / gain(col, row))
        pha = self.zero_suppress(pha, threshold)
        # Array indexes in order of decreasing pha---note that we use -pha to
        # trick argsort into sorting values in decreasing order.
        idx = np.argsort(-pha)
        # Only pick the seed and the N highest pixels.
        # This is useless for the circular readout because in that case all
        # neighbors are used for track reconstruction.
        mask = idx[:self.num_neighbors + 1]
        # Sort the arrays in decreasing order before applying the position suppression.
        pha, col, row = self.position_suppress(pha[mask], col[mask], row[mask])
        x, y = self.readout.pixel_to_world(col, row)
        return Cluster(x, y, col, row, pha, adc_to_ev, self.pos_recon_algorithm, self.recon_pars)


@dataclass
class ClusteringHex(ClusteringBase):

    """Hexagonal clustering.

    This clustering strategy always takes the six neighbors of the seed pixel, without applying
    any position suppression. The order of the pixels is fixed, with the seed pixel always in the
    first position, and the neighbors ordered clockwise, depending on the readout geometry.

    Arguments
    ---------
    pos_recon_algorithm : str
        The position reconstruction algorithm to use for the cluster position reconstruction.
        Possible values are "centroid" and "mle".
    recon_pars : dict, optional
        Dictionary containing the parameters for the position reconstruction algorithm.
    """

    pos_recon_algorithm: str = "mle"
    recon_pars: dict = None

    def run(self, event) -> Optional[Cluster]:
        """Overladed method.
        """
        # Load the readout calibration matrices.
        noise = self.readout.enc
        pedestal = self.readout.pedestal
        gain = self.readout.gain
        # Load the adc_to_ev conversion factor from the readout metadata of the
        # equalization matrix.
        adc_to_ev = gain.metadata["adc_to_ev"]
        if isinstance(event, DigiEventCircular):
            # Check if the seed pixel is at the border, in that case we throw away
            # the event.
            seed_coords = (event.column, event.row)
            if self.readout.is_at_border(*seed_coords):
                return None
            # Taking the NN logical coordinates ...
            neigh_coords = self.readout.neighbors(*seed_coords)
            col, row = np.vstack((seed_coords, neigh_coords)).T
            # ... transforming the coordinates in the corresponding ADC channel ...
            adc_channel_order = self.readout.adc_channel(col, row)
            # ... reordering the pha array for the correspondance (col[i], row[i]) with pha[i]
            # and applying pedestal and gain correction.
            pha = (event.pha[adc_channel_order] - pedestal(col, row)) / gain(col, row)
        elif isinstance(event, DigiEventRectangular):
            seed_coords = event.highest_pixel()
            # Check if the seed pixel is at the border, in that case we throw away the event.
            if self.readout.is_at_border(*seed_coords):
                return None
            neigh_coords = self.readout.neighbors(*seed_coords)
            col, row = np.vstack((seed_coords, neigh_coords)).T
            pha = (event(col, row) - pedestal(col, row)) / gain(col, row)
        else:
            raise RuntimeError(f"Unsupported event type {type(event)} for clustering")
        # Zero suppressing the event (whatever the readout type)...
        threshold = self.zero_sup_threshold * (noise(col, row) / gain(col, row))
        pha = self.zero_suppress(pha, threshold)
        # Calculate the physical coordinates of the pixels in the cluster.
        x, y = self.readout.pixel_to_world(col, row)
        return Cluster(x, y, col, row, pha, adc_to_ev, self.pos_recon_algorithm, self.recon_pars)
