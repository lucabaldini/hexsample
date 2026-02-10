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

"""Display facilities.
"""

from typing import Tuple

import matplotlib
import numpy as np
from aptapy.plotting import plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon

from .clustering import ClusteringNN
from .digi import DigiEventBase, DigiEventCircular, DigiEventRectangular
from .hexagon import HexagonalGrid
from .mc import MonteCarloEvent
from .readout import HexagonalReadoutBase
from .roi import RegionOfInterest


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
        kwargs.setdefault("edgecolor", "gray")
        kwargs.setdefault("facecolor", "none")
        kwargs.setdefault("linewidth", 1.2)
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
        self.color_map = matplotlib.colormaps[kwargs.get("cmap_name", "Reds")].copy()
        self.color_map_offset = kwargs.get("cmap_offset", 0)
        self.color_map.set_under("white")

    @staticmethod
    def setup_gca():
        """Setup the current axes object to make the display work.
        """
        plt.gca().set_aspect("equal")
        plt.gca().autoscale()
        plt.axis("off")

    @staticmethod
    def show():
        """Convenience function to setup the matplotlib canvas for an event display.
        """
        HexagonalGridDisplay.setup_gca()
        plt.show()

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
            fmt = dict(ha="center", va="center", size="xx-small")
            for (_x, _y, _col, _row) in zip(x, y, col, row):
                plt.text(_x + dx, _y + dy, f"({_col}, {_row})", **fmt)
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
            color = np.full(col.shape, "#555")
            color[~roi.in_rot(col, row)] = "#CCC"
            collection.set_edgecolor(color)
        plt.gca().add_collection(collection)
        # And if we want the indices, we add appropriate text patches.
        if indices:
            font_size = "x-small"
            cols, rows = roi.col_indexes(), roi.row_indexes()
            first_row = np.full(cols.shape, roi.min_row)
            first_col = np.full(rows.shape, roi.min_col)
            fmt = dict(fontsize=font_size, ha="center", va="bottom", rotation=60.)
            for x, y, col in zip(*self._grid.pixel_to_world(cols, first_row), cols):
                plt.text(x + dx, y + dy + self._grid.secondary_pitch, f"{col}", **fmt)
            fmt = dict(fontsize=font_size, ha="right", va="center", rotation=0.)
            for x, y, row in zip(*self._grid.pixel_to_world(first_col, rows), rows):
                plt.text(x + dx - self._grid.pitch, y + dy, f"{row}", **fmt)
        return collection

    def draw_digi_event_rectangular(self, event: DigiEventRectangular,
        offset: Tuple[float, float] = (0., 0.),
        indices: bool = True, padding: bool = True, zero_sup_threshold: float = 0,
        values: bool = True, **kwargs) -> HexagonCollection:
        """Draw an actual event int the parent hexagonal grid.

        This is taking over where the draw_roi() hook left, and adding the
        event part.
        """
        # pylint: disable = invalid-name, too-many-arguments, too-many-locals
        collection = self.draw_roi(event.roi, offset, indices, padding, **kwargs)
        if values:
            # Draw the pixel values
            fmt = dict(ha="center", va="center", fontsize="small")
            for x, y, value in zip(collection.x, collection.y, event.pha.flatten()):
                if value > zero_sup_threshold:
                    plt.text(x, y, f"{value}", color="black", **fmt)
        return collection

    def draw_digi_event_circular(self, event: DigiEventCircular,
        offset: Tuple[float, float] = (0., 0.), zero_sup_threshold: float = 0,
        values: bool = True, **kwargs) -> HexagonCollection:
        """Display a digi event with circular readout.
        """
        dx, dy = offset
        # This is shamelessly copied from clustering.py, and we should really
        # have a function in event that is returning the physical coordinates
        # and the pha values of all the pixels involved in the event.
        col = [event.column]
        row = [event.row]
        adc_channel_order = [self._grid.adc_channel(event.column, event.row)]
        # Taking the NN in logical coordinates ...
        for _col, _row in self._grid.neighbors(event.column, event.row):
            col.append(_col)
            row.append(_row)
            # ... transforming the coordinates of the NN in its corresponding ADC channel ...
            adc_channel_order.append(self. _grid.adc_channel(_col, _row))
        # ... reordering the pha array for the correspondence (col[i], row[i]) with pha[i].
        pha = event.pha[adc_channel_order]
        # Converting lists into numpy arrays
        cols = np.array(col)
        rows = np.array(row)
        pha = np.array(pha)
        x, y = self._grid.pixel_to_world(cols, rows)
        args = x + dx, y + dy, 0.5 * self._grid.pitch, self._grid.hexagon_orientation()
        collection = HexagonCollection(*args, **kwargs)
        if values:
            # Draw the pixel values
            fmt = dict(ha="center", va="center", fontsize="small")
            for x, y, value in zip(collection.x, collection.y, pha.flatten()):
                if value > zero_sup_threshold:
                    plt.text(x, y, f"{value}", color="black", **fmt)
        plt.gca().add_collection(collection)
        return collection

    def draw_digi_event(self, event, zero_sup_threshold) -> HexagonCollection:
        """Draw a digi event.

        This is just dispatching the call to the proper method depending
        on the event type.
        """
        if isinstance(event, DigiEventRectangular):
            return self.draw_digi_event_rectangular(event, zero_sup_threshold=zero_sup_threshold)
        if isinstance(event, DigiEventCircular):
            return self.draw_digi_event_circular(event, zero_sup_threshold=zero_sup_threshold)
        raise NotImplementedError(f"Cannot draw event of type {type(event)}.")

    def draw_positions(self, mc_event: MonteCarloEvent, digi_event: DigiEventBase,
                       readout: HexagonalReadoutBase, recon_defaults: object,
                       zero_sup_threshold: int) -> None:
        """Draw the Monte Carlo truth position and the reconstructed positions on top of the digi
        event.
        """
        # Plot the Monte Carlo truth position.
        plt.scatter(mc_event.absx, mc_event.absy, marker=".", s=100, label="Monte Carlo")
        # Calculate the cluster from the digi event.
        cluster = ClusteringNN(readout, zero_sup_threshold,
                               num_neighbors=6).run(digi_event)
        # Calculate and plot centroid position.
        centroid_position = cluster.centroid()
        plt.scatter(*centroid_position, marker="x", s=100, label="Centroid")
        # Calculate and plot eta reconstructed position.
        eta_recon_args = (recon_defaults.eta_2pix_rad, recon_defaults.eta_2pix_pivot,
                          recon_defaults.eta_3pix_rad0, recon_defaults.eta_3pix_rad1,
                          recon_defaults.eta_3pix_rad_pivot, recon_defaults.eta_3pix_theta0)
        try:
            eta_position = cluster.eta(*eta_recon_args, pitch=readout.pitch)
            # If cluster size is not 2 or 3, eta returns the centroid position, so we only
            # plot it if it's different from the centroid.
            plt.scatter(*eta_position, marker="+", s=100, label=r"$\eta$")
        except RuntimeError:
            pass
        plt.legend()
