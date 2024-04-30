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

from typing import Tuple

import matplotlib
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
import numpy as np

from hexsample.digi import DigiEventRectangular
from hexsample.hexagon import HexagonalGrid
from hexsample.plot import plt
from hexsample.roi import RegionOfInterest



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
    """

    def __init__(self, x, y, radius: float, orientation: float = 0., **kwargs) -> None:
        """Constructor.
        """
        # pylint: disable = invalid-name
        self.x = x
        self.y = y
        kwargs.setdefault('edgecolor', 'gray')
        kwargs.setdefault('facecolor', 'none')
        kwargs.setdefault('linewidth', 1.2)
        patches = [RegularPolygon(xy, 6, radius=radius, orientation=orientation) \
            for xy in zip(x, y)]
        # match_original is explicitely set to false so that new colors may be
        # assigned to individual members by providing the standard collection
        # arguments: facecolor, edgecolor, linewidths, norm or cmap.
        super().__init__(patches, match_original=False, **kwargs)



class HexagonalGridDisplay:

    """Display for an HexagonalGrid object.
    """

    def __init__(self, grid: HexagonalGrid, **kwargs) -> None:
        """Constructor.
        """
        self._grid = grid
        self.color_map = matplotlib.cm.get_cmap(kwargs.get('cmap_name', 'Reds')).copy()
        self.color_map_offset = kwargs.get('cmap_offset', 0)
        self.color_map.set_under('white')

    @staticmethod
    def setup_gca():
        """Setup the current axes object to make the display work.
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

    def pha_to_colors(self, pha: np.array, zero_sup_threshold: float = None) -> np.array:
        """Convert the pha values to colors for display purposes.
        """
        values = pha.flatten()
        values += self.color_map_offset
        if zero_sup_threshold is not None:
            values[values <= zero_sup_threshold + self.color_map_offset] = -1.
        values = values / float(values.max())
        return self.color_map(values)

    @staticmethod
    def brightness(color: np.array) -> np.array:
        """Quick and dirty proxy for the brighness of a given array of colors.

        See https://stackoverflow.com/questions/9733288
        and also
        https://stackoverflow.com/questions/30820962
        for how to split in columns the array of colors.
        """
        # pylint: disable = invalid-name
        r, g, b, _ = color.T
        return (299 * r + 587 * g + 114 * b) / 1000

    def draw(self, offset: Tuple[float, float] = (0., 0.), pixel_labels: bool = False,
        **kwargs) -> HexagonCollection:
        """Draw the full grid display.
        """
        # pylint: disable = invalid-name, too-many-locals
        col, row, x, y = self._grid.pixel_physical_coordinates()
        dx, dy = offset
        collection = HexagonCollection(x + dx, y + dy, 0.5 * self._grid.pitch,
            self._grid.hexagon_orientation(), **kwargs)
        plt.gca().add_collection(collection)
        if pixel_labels:
            fmt = dict(ha='center', va='center', size='xx-small')
            for (_x, _y, _col, _row) in zip(x, y, col, row):
                plt.text(_x + dx, _y + dy, f'({_col}, {_row})', **fmt)
        return collection

    def draw_roi(self, roi: RegionOfInterest, offset: Tuple[float, float] = (0., 0.),
        indices: bool = True, padding: bool = True, **kwargs) -> HexagonCollection:
        """Draw a given ROI.
        """
        # pylint: disable = invalid-name, too-many-locals
        # Calculate the coordinates of the pixel centers and build the hexagon collection.
        col, row = roi.serial_readout_coordinates()
        dx, dy = offset
        x, y = self._grid.pixel_to_world(col, row)
        args = x + dx, y + dy, 0.5 * self._grid.pitch, self._grid.hexagon_orientation()
        collection = HexagonCollection(*args, **kwargs)
        # If the padding is defined, we want to distinguish the different regions
        # by the pixel edge color.
        if padding:
            color = np.full(col.shape, '#555')
            color[~roi.in_rot(col, row)] = '#CCC'
            collection.set_edgecolor(color)
        plt.gca().add_collection(collection)
        # And if we want the indices, we add appropriate text patches.
        if indices:
            font_size = 'x-small'
            cols, rows = roi.col_indexes(), roi.row_indexes()
            first_row = np.full(cols.shape, roi.min_row)
            first_col = np.full(rows.shape, roi.min_col)
            fmt = dict(fontsize=font_size, ha='center', va='bottom', rotation=60.)
            for x, y, col in zip(*self._grid.pixel_to_world(cols, first_row), cols):
                plt.text(x + dx, y + dy + self._grid.secondary_pitch, f'{col}', **fmt)
            fmt = dict(fontsize=font_size, ha='right', va='center', rotation=0.)
            for x, y, row in zip(*self._grid.pixel_to_world(first_col, rows), rows):
                plt.text(x + dx - self._grid.pitch, y + dy, f'{row}', **fmt)
        return collection

    def draw_digi_event(self, event: DigiEventRectangular, offset: Tuple[float, float] = (0., 0.),
        indices: bool = True, padding: bool = True, zero_sup_threshold: float = 0,
        values: bool = True, **kwargs) -> HexagonCollection:
        """Draw an actual event int the parent hexagonal grid.

        This is taking over where the draw_roi() hook left, and adding the
        event part.
        """
        # pylint: disable = invalid-name, too-many-arguments, too-many-locals
        collection = self.draw_roi(event.roi, offset, indices, padding, **kwargs)
        face_color = self.pha_to_colors(event.pha, zero_sup_threshold)
        collection.set_facecolor(face_color)
        if values:
            # Draw the pixel values---note that we use black or white for the text
            # color depending on the brightness of the pixel.
            black = np.array([0., 0., 0., 1.])
            white = np.array([1., 1., 1., 1.])
            text_color = np.tile(black, len(face_color)).reshape(face_color.shape)
            text_color[self.brightness(face_color) < 0.5] = white
            fmt = dict(ha='center', va='center', fontsize='xx-small')
            for x, y, value, color in zip(collection.x, collection.y,\
                event.pha.flatten(), text_color):
                if value > zero_sup_threshold:
                    plt.text(x, y, f'{value}', color=color, **fmt)
        return collection
