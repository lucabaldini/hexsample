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


from enum import Enum, auto

import numpy as np


class HexagonalLayout(Enum):

    """Enum class expressing the possible grid layouts.
    """

    # Horizontal, pointy top, odd rows are shoved right.
    ODD_R = auto()
    # Horizontal, pointy top, even rows are shoved right.
    EVEN_R = auto()
    # Vertical, flat top, odd columns are shoved down.
    ODD_Q = auto()
    # Vertical, flat top, even columns are shoved down.
    EVEN_Q = auto()



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

    def __init__(self, layout : HexagonalLayout, num_cols : int, num_rows : int,
                 pitch : float) -> None:
        """Constructor.
        """
        self.layout = layout
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.num_pixels = self.num_cols * self.num_rows
        self.pitch = pitch
        self._secondary_pitch = 0.5 * np.sqrt(3.) * self.pitch
        self._hexagon_size = self.pitch / np.sqrt(3.)
        # Definition of the origin of coordinates.
        if self.pointy_topped():
            self.xoffset = 0.5 * (self.num_cols - 1 - 0.5 * self._parity_offset(1)) * self.pitch
            self.yoffset = 0.5 * (self.num_rows - 1) * self._secondary_pitch
        else:
            self.xoffset = 0.5 * (self.num_cols - 1) * self._secondary_pitch
            self.yoffset = 0.5 * (self.num_rows - 1 - 0.5 * self._parity_offset(1)) * self.pitch

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

    def _parity_offset(self, index : int) -> int:
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

    def pixel_to_world(self, col : np.array, row : np.array) -> tuple[np.array, np.array]:
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
            y = self.yoffset - row * self._secondary_pitch
        else:
            x = col * self._secondary_pitch - self.xoffset
            y = self.yoffset - (row - 0.5 * self._parity_offset(col)) * self.pitch
        return x, y

    def _float_axial(self, x : np.array, y : np.array) -> tuple[np.array, np.array]:
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
    def _axial_round(q : np.array, r : np.array) -> tuple[np.array, np.array]:
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
        dq = np.round(q + 0.5 * r) * (q**2. >= r**2.)
        dr = np.round(r + 0.5 * q) * (q**2. < r**2.)
        return (qgrid + dq).astype(int), (rgrid + dr).astype(int)

    def _axial_to_offset(self, q : np.array, r : np.array) -> tuple[np.array, np.array]:
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

    def world_to_pixel(self, x : np.array, y : np.array) -> tuple[np.array, np.array]:
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

    def __str__(self):
        """String formatting.
        """
        return f'{self.num_cols}x{self.num_rows} {self.layout.name} '\
            f'{self.__class__.__name__} @ {self.pitch} mm pitch'
