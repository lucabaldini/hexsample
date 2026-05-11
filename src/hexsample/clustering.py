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

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from aptapy.models import Probit

from .digi import DigiEventCircular, DigiEventRectangular
from .readout import HexagonalReadoutBase


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

    def calculate_eta(self) -> np.ndarray:
        """Return the eta values of the pixels in the cluster.
        """
        return np.array([_pha / self.pulse_height() for _pha in self.pha[1:]])

    def versors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the versors u and v for the cluster. Their definitions depend on the cluster size.
        For 2-pixel clusters u is the versor that points from the center of the pixel with the
        highest pha to the center of the other one, while v is the versor perpendicular to u in
        counterclockwise direction. For 3-pixel clusters u points from the center
        of the pixel with the highest pha to the midpoint of the line that connects the centers of
        the other two pixels, and v is perpendicular to u and points towards the second most
        charged pixel.

        Returns
        -------
        u : np.ndarray
            The u versor.
        v : np.ndarray
            The v versor.
        """
        if self.x.shape[0] == 2:
            u = np.array([self.x[1] - self.x[0], self.y[1] - self.y[0]])
            v = np.array([-u[1], u[0]])
        elif self.x.shape[0] == 3:
            u = np.array([self.x[1] + self.x[2] - 2 * self.x[0],
                          self.y[1] + self.y[2] - 2 * self.y[0]])
            v = np.array([-u[1], u[0]])
            if (self.x[1] - self.x[0]) * v[0] + (self.y[1] - self.y[0]) * v[1] < 0:
                v = -v
        else:
            raise RuntimeError("Cluster must contain 2 or 3 pixels to calculate versors")
        # It can happen that the versor is [0, 0] for events with strange geometries.
        # In that case we avoid NaN by setting the versor to [0, 0].
        with np.errstate(invalid="ignore"):
            norm = np.sqrt(np.sum(u**2))
            if norm > 0:
                u = u / norm
                v = v / norm
            else:
                u = np.zeros(2)
                v = np.zeros(2)
        return u, v

    def eta(self, eta_2pix_rad_sigma: float, eta_2pix_rad_pivot: float, eta_3pix_rad_offset: float,
            eta_3pix_rad_sigma: float, eta_3pix_rad_pivot: float, eta_3pix_theta_sigma: float,
            pitch: float) -> Tuple[float, float]:
        """Return the cluster reconstructed position using the eta function calibrated for 2
        and 3 pixel clusters. If cluster size is not 2 or 3, reconstruct the position with the
        centroid.

        Arguments
        ---------
        eta_2pix_rad_sigma : float
            Probit function sigma parameter for two pixel events.
        eta_2pix_rad_pivot : float
            Transition value from linear (0 to pivot) to probit (> pivot) for two pixel events.
        eta_3pix_rad_offset : float
            Probit function offset parameter for three pixel events radial position component.
        eta_3pix_rad_sigma : float
            Probit function sigma parameter for three pixel events radial position component.
        eta_3pix_rad_pivot : float
            Transition value from linear (0 to pivot) to probit (> pivot) for three pixel events
            radial position component.
        eta_3pix_theta_sigma : float
            Probit function sigma parameter for three pixel events angular position component.
        pitch : float
            The pitch of the pixels.
        """
        # Return the centroid position if it's not possible to use the eta function
        if self.size() not in (2, 3):
            return self.centroid()
        # Calculate versors and eta.
        u, v = self.versors()
        _eta = self.calculate_eta()

        if self.size() == 2:
            # For 2-pixel events we estimate the position along the line that connects the
            # two pixels using the probit function.
            if _eta[0] > eta_2pix_rad_pivot or eta_2pix_rad_pivot <= 0.:
                r = Probit().evaluate(_eta[0], 0.5, eta_2pix_rad_sigma)
            else:
                y_pivot = Probit().evaluate(eta_2pix_rad_pivot, 0.5, eta_2pix_rad_sigma)
                r = y_pivot / eta_2pix_rad_pivot * _eta[0]
            x_recon = self.x[0] + r * pitch * u[0]
            y_recon = self.y[0] + r * pitch * u[1]
        elif self.size() == 3:
            # For 3-pixel events we estimate both r and theta using the probit function.
            eta_sum = _eta[0] + _eta[1]
            eta_diff = (_eta[0] - _eta[1]) / eta_sum
            if eta_sum > eta_3pix_rad_pivot or eta_3pix_rad_pivot <= 0.:
                r = Probit().evaluate(eta_sum, eta_3pix_rad_offset, eta_3pix_rad_sigma)
            else:
                y_pivot = Probit().evaluate(eta_3pix_rad_pivot, eta_3pix_rad_offset,
                                            eta_3pix_rad_sigma)
                r = y_pivot / eta_3pix_rad_pivot * eta_sum
            theta = Probit().evaluate((eta_diff + 1)/2, 0, eta_3pix_theta_sigma) / r
            # Reconstructing the position using r and theta
            x_recon = self.x[0] + r * pitch * (np.cos(theta) * u[0] + np.sin(theta) * v[0])
            y_recon = self.y[0] + r * pitch * (np.cos(theta) * u[1] + np.sin(theta) * v[1])
        else:
            # This condition should never be reached because of the check at the beginning of the
            # method, but it's here for safety.
            raise RuntimeError("Cluster must contain 2 or 3 pixels to reconstruct position with" \
                               " eta function")
        return x_recon, y_recon

    def position(self):
        """Return the cluster reconstructed position using the position reconstruction algorithm
        specified in the constructor.

        This method is a wrapper around the different position reconstruction algorithms. It checks
        the value of pos_recon_algorithm and calls the corresponding method. If the value of
        pos_recon_algorithm is not recognized, it raises an error.
        """
        if self.pos_recon_algorithm == "centroid":
            return self.centroid()
        if self.pos_recon_algorithm == "eta":
            if self.recon_pars is None:
                raise RuntimeError("Eta reconstruction algorithm requires recon_pars to be set.")
            return self.eta(**self.recon_pars)
        raise RuntimeError(f"Unknown position reconstruction method {self.pos_recon_algorithm}")


@dataclass
class ClusteringBase:

    """Base class for the clustering.
    """

    readout: HexagonalReadoutBase
    zero_sup_threshold: float

    def zero_suppress(self, array: np.ndarray) -> np.ndarray:
        """Zero suppress a generic array.
        """
        out = array.copy()
        out[out <= self.zero_sup_threshold] = 0
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

    def run(self, event) -> Cluster:
        """Overladed method.

        .. warning::
           The loop ever the neighbors might likely be vectorized and streamlined
           for speed using proper numpy array for the offset indexes.
        """
        readout = self.readout
        if isinstance(event, DigiEventCircular):
            # If the readout is circular, we want to take all the neirest neighbors.
            # Trailing -1 is bc the central px is already considered.
            self.num_neighbors = 6 #HexagonalReadoutCircular.NUM_PIXELS - 1
            col = [event.column]
            row = [event.row]
            adc_channel_order = [readout.adc_channel(event.column, event.row)]
            # Taking the NN in logical coordinates ...
            for _col, _row in readout.neighbors(event.column, event.row):
                col.append(_col)
                row.append(_row)
                # ... transforming the coordinates of the NN in its corresponding ADC channel ...
                adc_channel_order.append(readout.adc_channel(_col, _row))
            # Converting lists into numpy arrays.
            pha = np.array(event.pha[adc_channel_order])
            col = np.array(col)
            row = np.array(row)
            # Applying the pedestal subtraction and gain correction.
            pha = (pha - readout.pedestal(col, row)) / readout.gain(col, row)
        # pylint: disable = invalid-name
        elif isinstance(event, DigiEventRectangular):
            seed_col, seed_row = event.highest_pixel()
            if readout.is_at_border(seed_col, seed_row):
                return None
            col = [seed_col]
            row = [seed_row]
            for _col, _row in readout.neighbors(seed_col, seed_row):
                col.append(_col)
                row.append(_row)
            col = np.array(col)
            row = np.array(row)
            pha = np.array([
                (event(_col, _row) - readout.pedestal(_col, _row)) / readout.gain(_col, _row)
                for _col, _row in zip(col, row)
            ])
        # Zero suppressing the event (whatever the readout type)...
        pha = self.zero_suppress(pha)
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
        return Cluster(x, y, col, row, pha, readout.gain.metadata["adc_to_ev"], self.pos_recon_algorithm, self.recon_pars)
