# Copyright (C) 2023 luca.baldini@pi.infn.it
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
from typing import Tuple

import numpy as np
from aptapy.models import PowerLaw

from .digi import DigiEventCircular, DigiEventRectangular, DigiEventSparse
from .hexagon import HexagonalGrid
from .readout import HexagonalReadoutCircular


@dataclass
class Cluster:

    """Small container class describing a cluster.
    """

    # pylint: disable = invalid-name

    x: np.ndarray
    y: np.ndarray
    pha: np.ndarray

    def __post_init__(self) -> None:
        """Small cross check on the dimensions of the arrays passed in the constructor.
        """
        if not self.x.shape == self.y.shape == self.pha.shape:
            raise RuntimeError(f"Inconsistent arrays: x = {self.x}, y = {self.y}, pha = {self.pha}")

    def size(self) -> int:
        """Return the size of the cluster.
        """
        # Modify to np.count_nonzero(self.pha) for compatibility with neural network
        return self.x.size

    def pulse_height(self) -> float:
        """Return the total pulse height of the cluster.
        """
        return self.pha.sum()

    def centroid(self) -> Tuple[float, float]:
        """Return the cluster centroid.
        """
        return np.average(self.x, weights=self.pha), np.average(self.y, weights=self.pha)

    def calculate_eta(self) -> np.ndarray:
        """Return the eta values for the cluster.
        """
        eta = np.array([_pha / self.pulse_height() for _pha in self.pha[1:]])
        return eta

    def n_versor(self) -> np.ndarray:
        """Return the versor n for the cluster. Its definitions depends on the cluster size.
        For 2-pixel clusters it is the versor that points from the center of the pixel with the
        highest pha to the center of the other one. For 3-pixel clusters it points from the center
        of the pixel with the highest pha to the midpoint of the line that connects the centers of
        the other two pixels."""
        if self.x.shape[0] == 2:
            n = np.array([self.x[1] - self.x[0], self.y[1] - self.y[0]]).T
        elif self.x.shape[0] == 3:
            n = np.array([self.x[1] + self.x[2] - 2 * self.x[0],\
                          self.y[1] + self.y[2] - 2 * self.y[0]]).T
        else:
            raise RuntimeError('Cluster must contain 2 or 3 pixels to calculate n versor')
        # It can happen that the versor is [0, 0] for events with strange geometries.
        # In that case we avoid NaN by setting the versor to [0, 0].
        with np.errstate(invalid='ignore'):
            n = n / np.sqrt(np.sum(n**2))
            if np.any(np.isnan(n)):
                n = np.array([0., 0.])
        return n

    def eta(self, gamma: float, pitch: float) -> Tuple[float, float]:
        """Return the cluster reconstructed position using the eta function.
        
        Arguments
        ---------
        gamma : floar
            The index of the power law of the eta function.
        pitch : float
            The pitch of the pixels.
        """
        # We want to extend this method to events with multiple pixels
        if self.size() != 2:
            raise RuntimeError('Cluster must contain only 2 pixels to use the eta function')

        diff = np.array([np.diff(self.x)[0], np.diff(self.y)[0]])
        n = diff / pitch

        # Consider to create a separate method for this
        eta = self.pha[1] / self.pulse_height()
        r_fit = PowerLaw().evaluate(eta/0.5, 0.5, gamma)*pitch
        x_fit = self.x[0] + r_fit * n[0]
        y_fit = self.y[0] + r_fit * n[1]

        return x_fit, y_fit


@dataclass
class ClusteringBase:

    """Base class for the clustering.
    """

    grid: HexagonalGrid
    zero_sup_threshold: float

    def zero_suppress(self, array: np.ndarray) -> np.ndarray:
        """Zero suppress a generic array.
        """
        out = array.copy()
        out[out <= self.zero_sup_threshold] = 0
        return out

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
    """

    num_neighbors: int

    def run(self, event) -> Cluster:
        """Overladed method.

        .. warning::
           The loop ever the neighbors might likely be vectorized and streamlined
           for speed using proper numpy array for the offset indexes.
        """
        if isinstance(event, DigiEventSparse):
            pass
        elif isinstance(event, DigiEventCircular):
            # If the readout is circular, we want to take all the neirest neighbors.
            # Trailing -1 is bc the central px is already considered.
            self.num_neighbors = HexagonalReadoutCircular.NUM_PIXELS - 1
            col = [event.column]
            row = [event.row]
            adc_channel_order = [self.grid.adc_channel(event.column, event.row)]
            # Taking the NN in logical coordinates ...
            for _col, _row in self.grid.neighbors(event.column, event.row):
                col.append(_col)
                row.append(_row)
                # ... transforming the coordinates of the NN in its corresponding ADC channel ...
                adc_channel_order.append(self.grid.adc_channel(_col, _row))
            # ... reordering the pha array for the correspondance (col[i], row[i]) with pha[i].
            pha = event.pha[adc_channel_order]
            # Converting lists into numpy arrays
            col = np.array(col)
            row = np.array(row)
            pha = np.array(pha)
        # pylint: disable = invalid-name
        elif isinstance(event, DigiEventRectangular):
            seed_col, seed_row = event.highest_pixel()
            col = [seed_col]
            row = [seed_row]
            for _col, _row in self.grid.neighbors(seed_col, seed_row):
                col.append(_col)
                row.append(_row)
            col = np.array(col)
            row = np.array(row)
            pha = np.array([event(_col, _row) for _col, _row in zip(col, row)])
        # Zero suppressing the event (whatever the readout type)...
        pha = self.zero_suppress(pha)
        # Array indexes in order of decreasing pha---note that we use -pha to
        # trick argsort into sorting values in decreasing order.
        idx = np.argsort(-pha)
        # Only pick the seed and the N highest pixels.
        # This is useless for the circular readout because in that case all
        # neighbors are used for track reconstruction.
        mask = idx[:self.num_neighbors + 1]
        # If there's any zero left in the target pixels, get rid of it.
        mask = mask[pha[mask] > 0]
        # Trim the relevant arrays.
        col = col[mask]
        row = row[mask]
        pha = pha[mask]
        x, y = self.grid.pixel_to_world(col, row)
        return Cluster(x, y, pha)
