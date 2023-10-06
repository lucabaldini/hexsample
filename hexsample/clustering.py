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

import numpy as np

from hexsample.digi import DigiEvent


@dataclass
class Cluster:

    """Small container class describing a cluster.
    """

    x : np.ndarray
    y : np.ndarray
    pha : np.ndarray

    def pulse_height(self) -> float:
        """Return the total pulse height of the cluster.
        """
        return self.pha.sum()

    def centroid(self) -> tuple[float, float]:
        """Return the cluster centroid.
        """
        return np.average(self.x, weights=self.pha), np.average(self.y, weights=self.pha)


@dataclass
class ClusteringNN:

    """
    """

    zero_sup_threshold : int
    num_neighbors : int

    def run(event : DigiEvent) -> Cluster:
        """
        """
        pass



if __name__ == '__main__':
    #cluster = Cluster(np.full(10, 1.), np.full(10, 1.), np.full(10, 123))
    #print(cluster)
    #print(cluster.pulse_height(), cluster.centroid())
    from hexsample.io import DigiInputFile
    from hexsample.digi import Xpol3
    grid = Xpol3()
    clustering = ClusteringNN(0, 6)
    for event in DigiInputFile('/home/lbaldini/hexsampledata/hxsim.h5'):
        print(event.ascii())
        col, row = event.highest_pixel()
        cols = [col]
        rows = [row]
        pha = [event(col, row)]
        for _col, _row in grid.neighbors(col, row):
            cols.append(_col)
            rows.append(_row)
            pha.append(event(_col, _row))
        cols = np.array(cols)
        rows = np.array(rows)
        x, y = grid.pixel_to_world(cols, row)
        pha = np.array(pha)
        print(cols, rows, x, y, pha)
        input()
