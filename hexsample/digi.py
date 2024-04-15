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

"""Digi event structures.
"""

from dataclasses import dataclass
from typing import Tuple

from loguru import logger
import numpy as np

from hexsample.hexagon import HexagonalGrid, HexagonalLayout
from hexsample.pprint import AnsiFontEffect, ansi_format, space, line
from hexsample.roi import Padding, RegionOfInterest



@dataclass
class DigiEventBase:

    """Base class for a digitized event.

    This includes alll the timing information that is common to the different digitized
    event structure, as well as the PHA content of the pixels in the event, but no
    information about the physical location on the latter within the readout chip.
    It is responsibility of the derived classes to provide this information, in the
    way that is more conventient, depending on the particular readout strategy.

    Arguments
    ---------
    trigger_id : int
        The trigger identifier.

    seconds : int
        The integer part of the timestamp.

    microseconds : int
        The fractional part of the timestamp.

    pha : np.ndarray
        The pixel content of the event, in the form of a 1-dimensional array.
    """

    trigger_id: int
    seconds: int
    microseconds: int
    livetime: int
    pha: np.ndarray

    def __eq__(self, other) -> bool:
        """Overloaded comparison operator.
        """
        return (self.trigger_id, self.seconds, self.microseconds, self.livetime) == \
            (other.trigger_id, other.seconds, other.microseconds, other.livetime) and \
            np.allclose(self.pha, other.pha)

    def timestamp(self) -> float:
        """Return the timestamp of the event, that is, the sum of the second and
        microseconds parts of the DigiEvent contributions as a floating point number.
        """
        return self.seconds + 1.e-6 * self.microseconds

    @classmethod
    def from_digi(cls, *args, **kwargs):
        """Build an event object from a row in a digitized file.
        """
        raise NotImplementedError

    def ascii(self, *args, **kwargs):
        """Ascii representation of the event.
        """
        raise NotImplementedError



@dataclass
class DigiEventSparse(DigiEventBase):

    """Sparse digitized event.

    In this particular incarnation of a digitized event we have no ROI structure,
    nor any rule as to the shape or morphology of the particular set of pixels
    being read out. The event represent an arbitrary collection of pixels, and we
    carry over the arrays of their row and column identifiers, in the form of
    two arrays whose length must match that of the PHA values.

    Arguments
    ---------
    columns : np.ndarray
        The column identifier of the pixels in the event (must have the same length
        of the pha class member).

    rows : np.ndarray
        The row identifier of the pixels in the event (must have the same length
        of the pha class member).
    """

    columns: np.ndarray
    rows: np.ndarray

    def __post_init__(self) -> None:
        """Post-initialization code.
        """
        if not len(self.rows) == len(self.columns) == len(self.pha):
            raise RuntimeError(f'{self.__class__.__name__} has {len(self.rows)} rows'
                f', {len(self.columns)} columns, and {len(self.pha)} PHA values')

    def as_dict(self) -> dict:
        """Return the pixel content of the event in the form of a {(col, row): pha}
        value.

        This is useful to address the pixel content at a given coordinate. We refrain,
        for the moment, from implementing a __call__() hook on top of this, as the
        semantics would be fundamentally different from that implemented with a
        rectangular ROI, where the access happen in ROI coordinates, and not in chip
        coordinates, but we might want to come back to this.
        """
        return {(col, row): pha for col, row, pha in zip(self.columns, self.rows, self.pha)}

    def ascii(self, pha_width: int = 5) -> str:
        """Ascii representation.
        """
        pha_dict = self.as_dict()
        fmt = f'%{pha_width}d'
        cols = np.arange(self.columns.min(), self.columns.max() + 1)
        num_cols = cols[-1] - cols[0] + 1
        rows = np.arange(self.rows.min(), self.rows.max() + 1)
        big_space = space(2 * pha_width + 1)
        text = f'\n{big_space}'
        text += ''.join([fmt % col for col in cols])
        text += f'\n{big_space}+{line(pha_width * num_cols)}\n'
        for row in rows:
            text += f'    {fmt % row}  |'
            for col in cols:
                try:
                    pha = fmt % pha_dict[(col, row)]
                except KeyError:
                    pha = ' ' * pha_width
                text += pha
            text += f'\n{big_space}|\n'
        return text



@dataclass
class DigiEventRectangular(DigiEventBase):

    """Specialized class for a digitized event based on a rectangular ROI.

    This implements the basic legacy machinery of the XPOL-I and XPOL-III readout chips.
    """

    roi: RegionOfInterest

    def __post_init__(self) -> None:
        """Post-initialization code.

        Here we reshape the one-dimensional PHA array coming from the serial
        readout to the proper ROI shape for all subsequent operations.
        """
        try:
            self.pha = self.pha.reshape(self.roi.shape())
        except ValueError as error:
            logger.error(f'Error in {self.__class__.__name__} post-initializaion.')
            print(self.roi)
            print(f'ROI size: {self.roi.size}')
            print(f'pha size: {self.pha.size}')
            logger.error(error)

    def __eq__(self, other) -> bool:
        """Overloaded comparison operator.
        """
        return DigiEventBase.__eq__(self, other) and self.roi == other.roi

    @classmethod
    def from_digi(cls, row: np.ndarray, pha: np.ndarray):
        """Alternative constructor rebuilding an object from a row on a digi file.

        This is used internally when we access event data in a digi file, and
        we need to reassemble a DigiEvent object from a given row of a digi table.
        """
        # pylint: disable=too-many-locals
        trigger_id, seconds, microseconds, livetime, min_col, max_col, min_row, max_row,\
            pad_top, pad_right, pad_bottom, pad_left = row
        padding = Padding(pad_top, pad_right, pad_bottom, pad_left)
        roi = RegionOfInterest(min_col, max_col, min_row, max_row, padding)
        return cls(trigger_id, seconds, microseconds, livetime, pha, roi)

    def __call__(self, col: int, row: int) -> int:
        """Retrieve the pha content of the event for a given column and row.

        Internally this is subtracting the proper offset to the column and row
        indexes in order to convert from chip coordinates to indexes of the
        underlying PHA array. Note that, due to the way arrays are indexed in numpy,
        we do need to swap the column and the row.

        Arguments
        ---------
        col : int
            The column number (in chip coordinates).

        row : int
            The row number (in chip coordinates).
        """
        return self.pha[row - self.roi.min_row, col - self.roi.min_col]

    def highest_pixel(self, absolute: bool = True) -> Tuple[int, int]:
        """Return the coordinates (col, row) of the highest pixel.

        Arguments
        ---------
        absolute : bool
            If true, the absolute coordinates (i.e., those referring to the readout
            chip) are returned; otherwise the coordinates are intended relative
            to the readout window (i.e., they can be used to index the pha array).
        """
        # Note col and row are swapped, here, due to how the numpy array are indexed.
        # pylint: disable = unbalanced-tuple-unpacking
        row, col = np.unravel_index(np.argmax(self.pha), self.pha.shape)
        if absolute:
            col += self.roi.min_col
            row += self.roi.min_row
        return col, row

    def ascii(self, pha_width: int = 5):
        """Ascii representation.
        """
        fmt = f'%{pha_width}d'
        cols = self.roi.col_indexes()
        rows = self.roi.row_indexes()
        big_space = space(2 * pha_width + 1)
        text = f'\n{big_space}'
        text += ''.join([fmt % col for col in cols])
        text += f'\n{big_space}'
        text += ''.join([fmt % (col - self.roi.min_col) for col in cols])
        text += f'\n{big_space}+{line(pha_width * self.roi.num_cols)}\n'
        for row in rows:
            text += f'{fmt % row} {fmt % (row - self.roi.min_row)}|'
            for col in cols:
                pha = fmt % self(col, row)
                if not self.roi.in_rot(col, row):
                    pha = ansi_format(pha, AnsiFontEffect.FG_BRIGHT_BLUE)
                text += pha
            text += f'\n{big_space}|\n'
        text += f'{self.roi}\n'
        return text



@dataclass
class DigiEventCircular(DigiEventBase):

    """Circular digitized event.

    In this particular incarnation of a digitized event the ROI is built around
    a central pixel, that is the one corresponding to maximum PHA. The ROI is then
    always (except in border-pixel cases) composed by 7 pixels: the central one and
    its 6 neighbours.

    Arguments
    ---------
    column : int
        The column identifier of the maximum PHA pixel in the event in pixel
        coordinates.

    row : int
        The column identifier of the maximum PHA pixel in the event in pixel
        coordinates.

    layout : HexagonalLayout
        The layout of the hexagonal grid of the chip. In a circular digi event it is needed
        because the map of the coordinates of the neighbors of a central pixel depend on the
        layout of the hexagonal grid.
    """

    column: int
    row: int
    grid: HexagonalGrid

    def get_neighbors(self):
        """This function returns the offset coordinates of the signal pixel, that
        is the highest pixel of the event (as first element), and its neighbors
        (starting from the upper left, proceiding clockwisely). 
        The order of the coordinates corresponds in the arrays of pha to the right 
        value, in order to properly reconstruct the event.
        """
        # Identifying the 6 neighbours of the central pixel and saving the signal pixels
        # prepending the cooridnates of the highest one... 
        neighbors_coords = list(self.grid.neighbors(self.column, self.row)) #returns a list of tuples
        # ...and prepending the highest pha pixel to the list...
        neighbors_coords.insert(0, (self.column, self.row))
        # ...dividing column and row coordinates in different arrays...
        columns, rows = zip(*neighbors_coords)
        columns, rows = np.array(columns), np.array(rows)
        return columns, rows
    
    def as_dict(self) -> dict:
        """Return the pixel content of the event in the form of a {(col, row): pha}
        value.

        This is useful to address the pixel content at a given coordinate. We refrain,
        for the moment, from implementing a __call__() hook on top of this, as the
        semantics would be fundamentally different from that implemented with a
        rectangular ROI, where the access happen in ROI coordinates, and not in chip
        coordinates, but we might want to come back to this.
        """
        columns, rows = self.get_neighbors()
        return {(col, row): pha for col, row, pha in zip(columns, rows, self.pha)}
    

    def ascii(self, pha_width: int = 5) -> str:
        """Ascii representation.
        """
        columns, rows = self.get_neighbors()
        pha_dict = self.as_dict()
        fmt = f'%{pha_width}d'
        cols = np.arange(columns.min(), columns.max() + 1)
        num_cols = cols[-1] - cols[0] + 1
        rows = np.arange(rows.min(), rows.max() + 1)
        big_space = space(2 * pha_width + 1)
        text = f'\n{big_space}'
        text += ''.join([fmt % col for col in cols])
        text += f'\n{big_space}+{line(pha_width * num_cols)}\n'
        for row in rows:
            text += f'    {fmt % row}  |'
            for col in cols:
                try:
                    pha = fmt % pha_dict[(col, row)]
                except KeyError:
                    pha = ' ' * pha_width
                text += pha
            text += f'\n{big_space}|\n'
        return text
    
    
