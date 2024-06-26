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

"""Geometrical facilities on a hexagonal grid.
"""


from enum import Enum
from typing import Tuple

import numpy as np


class HexagonalLayout(Enum):

    """Enum class expressing the possible grid layouts.
    """

    # Horizontal, pointy top, odd rows are shoved right.
    ODD_R = 'ODD_R'
    # Horizontal, pointy top, even rows are shoved right.
    EVEN_R = 'EVEN_R'
    # Vertical, flat top, odd columns are shoved down.
    ODD_Q = 'ODD_Q'
    # Vertical, flat top, even columns are shoved down.
    EVEN_Q = 'EVEN_Q'



def neighbors_odd_r(col: int, row: int) -> tuple:
    """Return a tuple with the coordinates of the 6 neighbor pixel for a given
    pixel in a ODD_R hexagonal grid.

    Arguments
    ---------
    col : int
        The column index.

    row : int
        The row index.
    """
    parity = row % 2
    return (col + parity, row - 1), (col + 1, row), (col + parity, row + 1),\
        (col + parity - 1, row + 1), (col - 1, row), (col + parity - 1, row - 1)

def neighbors_even_r(col: int, row: int) -> tuple:
    """Return a tuple with the coordinates of the 6 neighbor pixel for a given
    pixel in a EVEN_R hexagonal grid.

    Arguments
    ---------
    col : int
        The column index.

    row : int
        The row index.
    """
    parity = row % 2
    return (col - parity + 1, row - 1), (col + 1, row), (col - parity + 1, row + 1),\
        (col - parity, row + 1), (col - 1, row), (col - parity, row - 1)

def neighbors_odd_q(col: int, row: int) -> tuple:
    """Return a tuple with the coordinates of the 6 neighbor pixel for a given
    pixel in a ODD_Q hexagonal grid.

    Arguments
    ---------
    col : int
        The column index.

    row : int
        The row index.
    """
    parity = col % 2
    return (col, row - 1), (col + 1, row + parity - 1), (col + 1, row + parity),\
        (col, row + 1), (col - 1, row + parity), (col - 1, row + parity - 1)

def neighbors_even_q(col: int, row: int) -> tuple:
    """Return a tuple with the coordinates of the 6 neighbor pixel for a given
    pixel in a EVEN_Q hexagonal grid.

    Arguments
    ---------
    col : int
        The column index.

    row : int
        The row index.
    """
    parity = col % 2
    return (col, row - 1), (col + 1, row - parity), (col + 1, row - parity + 1),\
        (col, row + 1), (col - 1, row - parity + 1), (col - 1, row - parity)


# Lookup table for the functions returning the tuple of the 6 neighbor pixels in
# a hexagonal grid with a given layout.
_NEIGHBORS_PROXY_DICT = {
    HexagonalLayout.ODD_R: neighbors_odd_r,
    HexagonalLayout.EVEN_R: neighbors_even_r,
    HexagonalLayout.ODD_Q: neighbors_odd_q,
    HexagonalLayout.EVEN_Q: neighbors_even_q,
}



_N_ADC_CHANNELS = 7
_ADC_SEQUENCE_EVEN = (0, 2, 5, 0, 3, 5, 1, 3, 6, 1, 4, 6, 2, 4)
_ADC_SEQUENCE_ODD = (0, 3, 5, 1, 3, 6, 1, 4, 6, 2, 4, 0, 2, 5)
_ADC_SEQUENCE_LENGTH = len(_ADC_SEQUENCE_EVEN)


def adc_channel_odd_r(col: int, row: int) -> int:
    """Transformation from offset coordinates (col, row) into 7-adc channel label,
    that is an int between 0 and 6, for ODD_R grid layout.

    Arguments
    ---------
    col: int
        column pixel logical coordinate
    row: int
        row pixel logical coordinate
    """
    start = _ADC_SEQUENCE_ODD[row % _ADC_SEQUENCE_LENGTH]
    index = (col + start) % _N_ADC_CHANNELS
    return index

def adc_channel_even_r(col: int, row: int) -> int:
    """Transformation from offset coordinates (col, row) into 7-adc channel label,
    that is an int between 0 and 6, for EVEN_R grid layout.

    Arguments
    ---------
    col: int
        column pixel logical coordinate
    row: int
        row pixel logical coordinate
    """
    start = _ADC_SEQUENCE_EVEN[row % _ADC_SEQUENCE_LENGTH]
    index = (col + start) % _N_ADC_CHANNELS
    return index

def adc_channel_odd_q(col: int, row: int) -> int:
    """Transformation from offset coordinates (col, row) into 7-adc channel label,
    that is an int between 0 and 6, for ODD_Q grid layout.

    Arguments
    ---------
    col: int
        column pixel logical coordinate
    row: int
        row pixel logical coordinate
    """
    start = _ADC_SEQUENCE_ODD[col % _ADC_SEQUENCE_LENGTH]
    index = (row + start) % _N_ADC_CHANNELS
    return index

def adc_channel_even_q(col: int, row: int) -> int:
    """Transformation from offset coordinates (col, row) into 7-adc channel label,
    that is an int between 0 and 6, for EVEN_Q grid layout.

    Arguments
    ---------
    col: int
        column pixel logical coordinate
    row: int
        row pixel logical coordinate
    """
    start = _ADC_SEQUENCE_EVEN[col % _ADC_SEQUENCE_LENGTH]
    index = (row + start) % _N_ADC_CHANNELS
    return index


# Lookup table for .
_ADC_PROXY_DICT = {
    HexagonalLayout.ODD_R: adc_channel_odd_r,
    HexagonalLayout.EVEN_R: adc_channel_even_r,
    HexagonalLayout.ODD_Q: adc_channel_odd_q,
    HexagonalLayout.EVEN_Q: adc_channel_even_q,
}




class HexagonalGrid:

    # pylint: disable = too-many-instance-attributes

    """Generic hexagonal grid, with the origin of the physical coordinate
    system at its center.

    Arguments
    ---------
    layout : HexagonalLayout
        The underlying hexagonal grid layout.

    num_cols : int
        The number of columns in the grid

    num_rows : int
        The number of rows in the grid

    pitch : float
        The grid pitch in mm.
    """

    def __init__(self, layout: HexagonalLayout, num_cols: int, num_rows: int,
                 pitch: float) -> None:
        """Constructor.
        """
        self.layout = layout
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.num_pixels = self.num_cols * self.num_rows
        self.pitch = pitch
        self.secondary_pitch = 0.5 * np.sqrt(3.) * self.pitch
        self._hexagon_size = self.pitch / np.sqrt(3.)
        # Definition of the origin of coordinates.
        if self.pointy_topped():
            self.xoffset = 0.5 * (self.num_cols - 1 - 0.5 * self._parity_offset(1)) * self.pitch
            self.yoffset = 0.5 * (self.num_rows - 1) * self.secondary_pitch
        else:
            self.xoffset = 0.5 * (self.num_cols - 1) * self.secondary_pitch
            self.yoffset = 0.5 * (self.num_rows - 1 - 0.5 * self._parity_offset(1)) * self.pitch
        # Cache the proper function to retrieve the neighbor pixels from the
        # lookup table---this is used, e.g., for the clustering.
        self.neighbors = _NEIGHBORS_PROXY_DICT[self.layout]
        self.adc_channel = _ADC_PROXY_DICT[self.layout]

    def pointy_topped(self) -> bool:
        """Return True if the layout is pointy-topped.
        """
        return self.layout in (HexagonalLayout.ODD_R, HexagonalLayout.EVEN_R)

    def flat_topped(self) -> bool:
        """Return True if the layout is flat-topped.
        """
        return self.layout in (HexagonalLayout.ODD_Q, HexagonalLayout.EVEN_Q)

    def even(self) -> bool:
        """Return True if the layout is even.
        """
        return self.layout in (HexagonalLayout.EVEN_R, HexagonalLayout.EVEN_Q)

    def odd(self) -> bool:
        """Return True if the layout is odd.
        """
        return self.layout in (HexagonalLayout.ODD_R, HexagonalLayout.ODD_Q)

    def hexagon_orientation(self) -> float:
        """Return the orientation (rotation angle in radians) of the grid hexagons.

        This is calculated according to the matplotlib conventions for a
        RegularPolygon, that is, it's 0 for a pointy top, and pi/2 for a flat top.
        """
        return 0. if self.pointy_topped() else 0.5 * np.pi

    def _parity_offset(self, index: int) -> int:
        """Small convenience function to help with the tranformation.

        For any given column or row index, this returns 0 if the index is even
        and +1 or - 1 (depending on whether the parent layout is even or odd)
        if the index is odd.

        Arguments
        ---------
        index : int
            The column or row index.
        """
        value = index & 1
        return value if self.even() else - value

    def pixel_to_world(self, col: np.array, row: np.array) -> Tuple[np.array, np.array]:
        """Transform pixel coordinates to world coordinates.

        Arguments
        ---------
        col : array_like
            The input column number(s).

        row : array_like
            The input row number(s).
        """
        # pylint: disable = invalid-name
        if self.pointy_topped():
            x = (col - 0.5 * self._parity_offset(row)) * self.pitch - self.xoffset
            y = self.yoffset - row * self.secondary_pitch
        else:
            x = col * self.secondary_pitch - self.xoffset
            y = self.yoffset - (row - 0.5 * self._parity_offset(col)) * self.pitch
        return x, y

    def _float_axial(self, x: np.array, y: np.array) -> Tuple[np.array, np.array]:
        """Conversion of a given set of world coordinates into fractional axial
        coordinates, a. k. a. step 1 in the transformation between world coordinates
        to pixel coordinates.

        See https://www.redblobgames.com/grids/hexagons/ for more details.

        Arguments
        ---------
        x : array_like
            The x coordinate(s).

        y : array_like
            The y coordinate(s).
        """
        # pylint: disable = invalid-name
        if self.pointy_topped():
            q = (x / np.sqrt(3.) - y / 3.) / self._hexagon_size
            r = 2. / 3. * y / self._hexagon_size
        else:
            q = 2. / 3. * x / self._hexagon_size
            r = (-x / 3. + y / np.sqrt(3.)) / self._hexagon_size
        return q, r

    @staticmethod
    def _axial_round(q: np.array, r: np.array) -> Tuple[np.array, np.array]:
        """Rounding to integer of the axial coordinates, a. k. a. step 2 in the
        transformation between world coordinates to pixel coordinates.

        See https://www.redblobgames.com/grids/hexagons/ for more details.

        Arguments
        ---------
        q : array_like
            The q axial coordinate(s).

        r : array_like
            The r axial coordinate(s).
        """
        # pylint: disable = invalid-name
        qgrid = np.round(q)
        rgrid = np.round(r)
        q -= qgrid
        r -= rgrid
        mask = np.abs(q) >= np.abs(r)
        dq = np.round(q + 0.5 * r) * mask
        dr = np.round(r + 0.5 * q) * np.logical_not(mask)
        q = (qgrid + dq).astype(int)
        r = (rgrid + dr).astype(int)
        return q, r

    def _axial_to_offset(self, q: np.array, r: np.array) -> Tuple[np.array, np.array]:
        """Conversion from axial to offset coordinates, a. k. a. step 3 in the
        transformation between world coordinates to pixel coordinates.

        See https://www.redblobgames.com/grids/hexagons/ for more details.

        Arguments
        ---------
        q : array_like
            The q axial coordinate(s).

        r : array_like
            The r axial coordinate(s).
        """
        # pylint: disable = invalid-name
        if self.pointy_topped():
            col = q + (r + self._parity_offset(r)) // 2
            row = r
        else:
            col = q
            row = r + (q + self._parity_offset(q)) // 2
        return col, row

    def world_to_pixel(self, x: np.array, y: np.array) -> Tuple[np.array, np.array]:
        """Transform world coordinates to pixel coordinates.

        This proceeds in three basic steps (conversion to fractional axial coordinates,
        rounding to integer and conversion to offset coordinates) as described in
        https://www.redblobgames.com/grids/hexagons/ and we factored the three
        steps into three separate functions to clarify the code flow.

        Arguments
        ---------
        x : array_like
            The input x coordinate(s).

        y : array_like
            The input y coordinate(s).
        """
        # pylint: disable = invalid-name
        # Add back the offsets---and remember the y axis is running from the
        # top to the bottom, hence the different sign. Pay attention you cannot
        # do x += offset, here, as you would change the array in place.
        x = x + self.xoffset
        y = -y + self.yoffset
        # Calculate the "fractional" axial coordinates...
        q, r = self._float_axial(x, y)
        # ... round to integer...
        q, r = self._axial_round(q, r)
        # ... and convert to offset coordinates.
        col, row = self._axial_to_offset(q, r)
        return col, row

    def pixel_logical_coordinates(self) -> Tuple[np.array, np.array]:
        """Return a 2-element tuple (cols, rows) of numpy arrays containing all the
        column and row indices that allow to loop over the full matrix. The specific
        order in which the pixels are looped upon is completely arbitrary and, for
        the sake of this function, we assume that, e.g., for a 2 x 2 grid we return
        >>> cols = [0, 1, 0, 1]
        >>> rows = [0, 0, 1, 1]
        i.e., we loop with the following order
        >>> (0, 0), (1, 0), (0, 1), (1, 1)
        """
        cols = np.tile(np.arange(self.num_cols), self.num_rows)
        rows = np.repeat(np.arange(self.num_rows), self.num_cols)
        return cols, rows

    def pixel_physical_coordinates(self) -> Tuple[np.array, np.array, np.array, np.array]:
        """Return a 4-element tuple (cols, rows, x, y) containing the logical
        coordinates returned by pixel_logical_coordinates(), along with the
        corresponding physical coordinates.
        """
        cols, rows = self.pixel_logical_coordinates()
        x, y = self.pixel_to_world(cols, rows)
        return cols, rows, x, y

    def __str__(self):
        """String formatting.
        """
        return f'{self.num_cols}x{self.num_rows} {self.layout.name} '\
            f'{self.__class__.__name__} @ {self.pitch} mm pitch'
