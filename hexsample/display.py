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

"""Display facilities.
"""


from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
import numpy as np

from hexsample.hexagon import HexagonalGrid
from hexsample.plot import plt



class HexagonCollection(PatchCollection):

    """Collection of native matplotlib hexagon patches.

    Arguments
    ---------
    x : array_like
        The x coordinates of the hexagon centers.

    y : array_like
        The y coordinates of the hexagon centers.

    radius : float
        The distance from the center to each of the hexagon vertices.

    orientation: float
        The hexagon orientation in radians---zero means pointy topped.

    kwargs
        The keyword arguments to be passed to the PatchCollection constructor.
    """

    def __init__(self, x, y, radius : float, orientation : float = 0., **kwargs) -> None:
        """Constructor.
        """
        # pylint: disable = invalid-name
        self.x = x
        self.y = y
        kwargs.setdefault('edgecolor', 'gray')
        kwargs.setdefault('facecolor', 'none')
        kwargs.setdefault('linewidth', 1.2)
        patches = [RegularPolygon(xy, 6, radius, orientation) for xy in zip(x, y)]
        # match_original is explicitely set to false so that new colors may be
        # assigned to individual members by providing the standard collection
        # arguments: facecolor, edgecolor, linewidths, norm or cmap.
        PatchCollection.__init__(self, patches, match_original=False, **kwargs)



class HexagonalGridDisplay:

    """Display for an HexagonalGrid object.
    """

    def __init__(self, grid : HexagonalGrid) -> None:
        """Constructor.
        """
        self._grid = grid

    @staticmethod
    def setup_gca():
        """
        """
        plt.gca().set_aspect('equal')
        plt.gca().autoscale()
        plt.axis('off')

    @staticmethod
    def show():
        """Convenience function to setup the matplotlib canvas for an event display.
        """
        HexagonalGridDisplay.setup_gca()
        plt.show()

    def draw(self, offset : tuple[float, float] = (0., 0.), pixel_labels : bool = False,
        **kwargs) -> HexagonCollection:
        """Draw the full grid display.
        """
        col = np.tile(np.arange(self._grid.num_cols), self._grid.num_rows)
        row = np.repeat(np.arange(self._grid.num_rows), self._grid.num_cols)
        x, y = self._grid.pixel_to_world(col, row)
        dx, dy = offset
        collection = HexagonCollection(x + dx, y + dy, 0.5 * self._grid.pitch,
            self._grid.hexagon_orientation(), **kwargs)
        plt.gca().add_collection(collection)
        if pixel_labels:
            fmt = dict(ha='center', va='center', size='xx-small')
            for (_x, _y, _col, _row) in zip(x, y, col, row):
                plt.text(_x + dx, _y + dy, f'({_col}, {_row})', **fmt)
        return collection
