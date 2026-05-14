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
from matplotlib.widgets import Button, TextBox

from hexsample.fileio import DigiInputFileBase

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

    def __init__(self, grid: HexagonalGrid, zero_sup_threshold: float = 0.,
                 **kwargs) -> None:
        """Constructor.
        """
        self._grid = grid
        self.zero_sup_threshold = zero_sup_threshold
        self.color_map = matplotlib.colormaps[kwargs.get("cmap_name", "Reds")].copy()
        self.color_map_offset = kwargs.get("cmap_offset", 0)
        self.color_map.set_under("white")
        self.recon_pars = kwargs.get("recon_pars")
        self.figure, self.axes = plt.gcf(), plt.gca()
        self.figure.subplots_adjust(bottom=0.2)

    def setup_gca(self):
        self.axes.set_aspect("equal")
        self.axes.autoscale()
        self.axes.axis("off")

    def show(self):
        """Convenience function to setup the matplotlib canvas for an event display.
        """
        self.setup_gca()
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
        self.axes.add_collection(collection)
        if pixel_labels:
            fmt = dict(ha="center", va="center", size="xx-small")
            for (_x, _y, _col, _row) in zip(x, y, col, row):
                self.axes.text(_x + dx, _y + dy, f"({_col}, {_row})", **fmt)
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
        self.axes.add_collection(collection)
        # And if we want the indices, we add appropriate text patches.
        if indices:
            font_size = "x-small"
            cols, rows = roi.col_indexes(), roi.row_indexes()
            first_row = np.full(cols.shape, roi.min_row)
            first_col = np.full(rows.shape, roi.min_col)
            fmt = dict(fontsize=font_size, ha="center", va="bottom", rotation=0.)
            for x, y, col in zip(*self._grid.pixel_to_world(cols, first_row), cols):
                self.axes.text(x + dx, y + dy + self._grid.secondary_pitch, f"{col}", **fmt)
            fmt = dict(fontsize=font_size, ha="right", va="center", rotation=0.)
            for x, y, row in zip(*self._grid.pixel_to_world(first_col, rows), rows):
                self.axes.text(x + dx - self._grid.pitch, y + dy, f"{row}", **fmt)
        return collection

    def draw_digi_event_rectangular(self, event: DigiEventRectangular,
        offset: Tuple[float, float] = (0., 0.),
        indices: bool = True, padding: bool = True, zero_sup_threshold: float = 0.,
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
                    self.axes.text(x, y, f"{value}", color="black", **fmt)
        return collection

    def draw_digi_event_circular(self, event: DigiEventCircular,
        offset: Tuple[float, float] = (0., 0.), zero_sup_threshold: float = 0.,
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
                    self.axes.text(x, y, f"{value}", color="black", **fmt)
        self.axes.add_collection(collection)
        return collection

    def draw_digi_event(self, event: DigiEventBase, zero_sup_threshold: int) -> HexagonCollection:
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
                       readout: HexagonalReadoutBase,
                       zero_sup_threshold: int) -> None:
        """Draw the Monte Carlo truth position and the reconstructed positions on top of the digi
        event.
        """
        # Plot the Monte Carlo truth position.
        self.axes.scatter(mc_event.absx, mc_event.absy, marker=".", s=100, label="Monte Carlo")
        # Calculate the cluster from the digi event.
        cluster = ClusteringNN(readout, zero_sup_threshold, num_neighbors=6,
                               pos_recon_algorithm="centroid", recon_pars=None).run(digi_event)
        # Calculate and plot centroid position.
        centroid_position = cluster.centroid()
        self.axes.scatter(*centroid_position, marker="x", s=100, label="Centroid")
        # Calculate and plot eta reconstructed position.
        try:
            eta_position = cluster.eta(**self.recon_pars)
            # If cluster size is not 2 or 3, eta returns the centroid position, so we only
            # plot it if it's different from the centroid.
            self.axes.scatter(*eta_position, marker="+", s=100, label=r"$\eta$")
        except RuntimeError:
            pass
        self.axes.legend()


class EventDisplay(HexagonalGridDisplay):

    """Class to display events from a Digi input file.

    If the keyword argument recon_pars is provided, the display will also show the reconstructed
    positions and the Monte Carlo truth position (if available) on top of the digi event.

    Arguments
    ---------
    input_file: DigiInputFileBase
        The input file to read the events from.
    
    grid: HexagonalGrid
        The grid to use for the display.
    """

    def __init__(self, input_file: DigiInputFileBase, grid: HexagonalGrid, **kwargs):
        """Class constructor."""
        super().__init__(grid, **kwargs)
        self._input_file = input_file
        self.event_id = 0
        ##Draw the previous and next buttons for event navigation
        axprev = self.figure.add_axes([0.7, 0.05, 0.12, 0.075])
        self.prev_button = Button(axprev, 'Previous')
        axnext = self.figure.add_axes([0.83, 0.05, 0.12, 0.075])
        self.next_button = Button(axnext, 'Next')
        # Draw the textbox for event ID input
        axbox = self.figure.add_axes([0.5, 0.05, 0.1, 0.075])
        self.event_id_text_box = TextBox(axbox, "Event ID: ", textalignment="center")
        # Textbox for zero suppression threshold input. With this we redraw the
        # event with the newly picked threshold value.
        axzerosup = self.figure.add_axes([0.25, 0.05, 0.1, 0.075])
        self.zero_sup_text_box = TextBox(axzerosup, "Zero Sup. Thresh.: ", textalignment="center")
        self.show()
        self.next(None)

    def setup_gca(self):
        """Setup the current axes object to make the display work.
        Includes a modified axes adding a text box for event ID input.
        """
        ### Assign the first event in the file as the first displayed event
        self.axes.set_aspect("equal")
        initial_event = self._input_file.pick_event(int(self.event_id))
        self.draw_digi_event(initial_event, self.zero_sup_threshold)
        if isinstance(self._grid, HexagonalReadoutBase):
            self.draw_positions(self._input_file.current_mc_event(), initial_event,
                                self._grid, self.zero_sup_threshold)
        self.axes.autoscale()
        self.axes.axis("off")
        self.prev_button.on_clicked(self.prev)
        self.next_button.on_clicked(self.next)
        self.event_id_text_box.on_submit(self.pick_event)
        # Show current event ID in the text box after picking the event.
        self.event_id_text_box.set_val(initial_event.trigger_id)
        self.zero_sup_text_box.on_submit(self.update_zero_sup)
        # Show current zero suppression threshold in the text box.
        self.zero_sup_text_box.set_val(self.zero_sup_threshold)

    def current_event_id(self) -> int:
        """Convenience method to get the current event ID.
        """
        return int(self.event_id_text_box.text)

    def current_zero_sup_threshold(self) -> int:
        """Convenience method to get the current zero suppression threshold.
        """
        return float(self.zero_sup_text_box.text)

    def _draw(self, event: DigiEventBase) -> None:
        """Complete draw method for an event.

        This is a private hook that is called by the event navigation methods
        (next, prev, pick_event) to redraw the underlying digi event, as well as
        the relevant reconstructed quantities.
        """
        self.axes.clear()
        self.draw_digi_event(event, self.zero_sup_threshold)
        if isinstance(self._grid, HexagonalReadoutBase):
            self.draw_positions(self._input_file.current_mc_event(), event, self._grid,
                                self.zero_sup_threshold)
        self.axes.autoscale()
        self.axes.axis("off")
        # Show current event ID in the text box after picking the event. This can be improved
        self.event_id_text_box.set_val(event.trigger_id)
        self.figure.canvas.draw_idle()

    def next(self, _) -> DigiEventBase:
        """Convenience method to get the next event from the input file.
        """
        self._draw(next(self._input_file))

    def prev(self, _) -> DigiEventBase:
        """Convenience method to get the previous event from the input file.
        """
        self._draw(self._input_file.prev())

    def pick_event(self, _) -> DigiEventBase:
        """Convenience method to get a specific event from the input file.
        """
        event = self._input_file.pick_event(self.current_event_id())
        self._draw(event)

    def update_zero_sup(self, _) -> DigiEventBase:
        """Convenience method to update the zero suppression threshold and re-plot the event.
        """
        self.zero_sup_threshold = self.current_zero_sup_threshold()
        event = self._input_file.pick_event(self.current_event_id())
        self._draw(event)

    def show(self):
        """Convenience function to setup the matplotlib canvas for an event display.
        """
        self.setup_gca()
        plt.show()
