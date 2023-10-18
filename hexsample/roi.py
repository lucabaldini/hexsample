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

"""Description of a region of interest.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np



@dataclass
class Padding:

    """Padding on the outside of the region of trigger.

    This is setup so that the padding can be initialized with a single argument,
    assuming that it is the same on all four sides, with two arguments
    (bottom = top and left = right), or with the padding different on all
    four sides.

    Arguments
    ---------
    top : int
        Top-side padding in pixels.

    right : int
        Right-side padding in pixels.

    bottom : int
        Bottom-side padding in pixels.

    left : int
        Left-side padding in pixels.
    """

    top : int
    right : int = None
    bottom : int = None
    left : int = None

    def __post_init__(self) -> None:
        """Overloaded dataclass method.
        """
        if self.right is None:
            self.right = self.top
        if self.bottom is None:
            self.bottom = self.top
        if self.left is None:
            self.left = self.right

    def __eq__(self, other):
        """Overloaded comparison operator.
        """
        return tuple(self) == tuple(other)

    def __iter__(self):
        """Make the class iterable, in the order (top, right, bottom, left)

        This is handy when looping over all the padding values, or to convert
        the padding into a tuple.
        """
        return iter((self.top, self.right, self.bottom, self.left))



@dataclass
class RegionOfInterest:

    """Class describing a region of interest (ROI).

    A region of interest is the datum of the logical coorinates of its two
    extreme corners, in the order (min_col, max_col, min_row, max_row).
    """

    # pylint: disable=too-many-instance-attributes

    min_col : int
    max_col : int
    min_row : int
    max_row : int
    padding : Padding

    def __post_init__(self) -> None:
        """Overloaded dataclass method.
        """
        self.num_cols = self.max_col - self.min_col + 1
        self.num_rows = self.max_row - self.min_row + 1
        self.size = self.num_cols * self.num_rows
        # Convenience hack to be able to pass an integer to the constructor, in
        # case the padding is the same on all four sides.
        if isinstance(self.padding, int):
            self.padding = Padding(self.padding)

    def __eq__(self, other):
        """Overloaded comparison operator.
        """
        return (self.min_col, self.max_col, self.min_row, self.max_row) == \
            (other.min_col, other.max_col, other.min_row, other.max_row) and\
            self.padding == other.padding

    def shape(self) -> Tuple[int, int]:
        """Return the shape of the ROI.

        Note that rows goes first and cols goes last---this is the shape that
        needs to be used to reshape the one-dimensional pha array so that
        it gets printed on the screen with the right size and orientation.
        """
        return self.num_rows, self.num_cols

    def at_border(self, chip_size : Tuple[int, int]):
        """Return True if the ROI is on the border for a given chip_size.

        We should consider making the chip size a class member, because it looks
        kind of awkward to pass it as an argument here.
        """
        num_cols, num_rows = chip_size
        return self.min_col == 0 or self.max_col == num_cols - 1 or\
            self.min_row == 0 or self.max_row == num_rows - 1

    def col_indexes(self) -> np.array:
        """Return an array with all the valid column indexes.
        """
        return np.arange(self.min_col, self.max_col + 1)

    def row_indexes(self) -> np.array:
        """Return an array with all the valid row indexes.
        """
        return np.arange(self.min_row, self.max_row + 1)

    def serial_readout_coordinates(self) -> Tuple[np.array, np.array]:
        """Return two one-dimensional arrays containing the column and row
        indexes, respectively, in order of serial readout of the ROI.

        Example
        -------
        >>> col, row = RegionOfInterest(0, 4, 0, 3).serial_readout_coordinates()
        >>> print(col)
        >>> [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        >>> print(row)
        >>> [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        """
        col = np.tile(self.col_indexes(), self.num_rows)
        row = np.repeat(self.row_indexes(), self.num_cols)
        return col, row

    def serial_readout_indexes(self) -> np.array:
        """Return a zero-indexed, two-dimensional array containing the serial
        readout index for each pixel in the ROI.

        Example
        -------
        >>> print(RegionOfInterest(0, 4, 0, 3).serial_readout_indexes())
        >>> [[ 0  1  2  3  4]
             [ 5  6  7  8  9]
             [10 11 12 13 14]
             [15 16 17 18 19]]
        """
        return np.arange(self.size).reshape(self.shape())

    def rot_slice(self) -> Tuple[slice, slice]:
        """Return a pair of slice objects that can be used to address the ROT
        part of a numpy array representing, e.g., the pha values of a given event.

        Note this operates in relative coordinates, i.e., native numpy indexing.
        """
        return slice(self.padding.top, self.num_rows - self.padding.bottom), \
            slice(self.padding.left, self.num_cols - self.padding.right)

    def rot_mask(self) -> np.array:
        """Return a two-dimensional, boolean array mask, with the same dimensions
        of the region of interest, that is True in the region of trigger and False
        in the outer, padding area.

        Note this operates in relative coordinates, i.e., native numpy indexing.
        """
        mask = np.zeros(self.shape(), dtype=bool)
        mask[self.rot_slice()] = True
        return mask

    def in_rot(self, col : np.array, row : np.array) -> np.array:
        """Return a boolean mask indicaing whether elements of the (col, row)
        arrays lie into the ROT area.
        """
        return np.logical_and.reduce((
            col >= self.min_col + self.padding.left,
            col <= self.max_col - self.padding.right,
            row >= self.min_row + self.padding.top,
            row <= self.max_row - self.padding.bottom
        ))
