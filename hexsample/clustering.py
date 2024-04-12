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

from hexsample.digi import DigiEventRectangular
from hexsample.hexagon import HexagonalGrid


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
            raise RuntimeError(f'Inconsistent arrays: x = {self.x}, y = {self.y}, pha = {self.pha}')

    def size(self) -> int:
        """Return the size of the cluster.
        """
        return self.x.size

    def pulse_height(self) -> float:
        """Return the total pulse height of the cluster.
        """
        return self.pha.sum()

    def centroid(self) -> Tuple[float, float]:
        """Return the cluster centroid.
        """
        return np.average(self.x, weights=self.pha), np.average(self.y, weights=self.pha)



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

    def run(self, event: DigiEventRectangular) -> Cluster:
        """Overladed method.

        .. warning::
           The loop ever the neighbors might likely be vectorized and streamlined
           for speed using proper numpy array for the offset indexes.
        """
        # pylint: disable = invalid-name
        seed_col, seed_row = event.highest_pixel()
        col = [seed_col]
        row = [seed_row]
        for _col, _row in self.grid.neighbors(seed_col, seed_row):
            col.append(_col)
            row.append(_row)
        col = np.array(col)
        row = np.array(row)
        pha = np.array([event(_col, _row) for _col, _row in zip(col, row)])
        pha = self.zero_suppress(pha)
        # Array indexes in order of decreasing pha---note that we use -pha to
        # trick argsort into sorting values in decreasing order.
        idx = np.argsort(-pha)
        # Only pick the seed and the N highest pixels.
        mask = idx[:self.num_neighbors + 1]
        # If there's any zero left in the target pixels, get rid of it.
        mask = mask[pha[mask] > 0]
        # Trim the relevant arrays.
        col = col[mask]
        row = row[mask]
        pha = pha[mask]
        x, y = self.grid.pixel_to_world(col, row)
        return Cluster(x, y, pha)
