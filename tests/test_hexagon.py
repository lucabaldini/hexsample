# Copyright (C) 2022 luca.baldini@pi.infn.it
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Test suite for hexagon.py
"""

import numpy as np

from hexsample.hexagon import HexagonalLayout, HexagonalGrid
from hexsample.display import HexagonalGridDisplay
from hexsample.plot import plt


def test_parity(nside : int = 10, pitch : float = 0.1):
    """Test the HexagonalGrid._parity_offset() method.
    """
    for layout in (HexagonalLayout.EVEN_R, HexagonalLayout.EVEN_Q):
        grid = HexagonalGrid(layout, nside, nside, pitch)
        assert grid._parity_offset(0) == 0
        assert grid._parity_offset(1) == 1
    for layout in (HexagonalLayout.ODD_R, HexagonalLayout.ODD_Q):
        grid = HexagonalGrid(layout, nside, nside, pitch)
        assert grid._parity_offset(0) == 0
        assert grid._parity_offset(1) == -1

def test_coordinate_transform(nside : int = 10, pitch : float = 0.1):
    """Simple test of the coordinate transformations: we pick the four corner
    pixels and verify that pixel_to_world() and word_to_pixel() roundtrip.
    (This not really an exahustive test, and all the points are at the center
    of pixels.)
    """
    test_pixels = ((0, 0), (0, nside - 1), (nside - 1, 0), (nside - 1, nside - 1))
    for layout in HexagonalLayout:
        grid = HexagonalGrid(layout, nside, nside, pitch)
        for col, row in test_pixels:
            x, y = grid.pixel_to_world(col, row)
            assert grid.world_to_pixel(x, y) == (col, row)

def test_display(nside : int = 10, pitch : float = 0.1):
    """Display all the four possible layout in a small arrangement.
    """
    target_col = 5
    target_row = 5
    for layout in HexagonalLayout:
        plt.figure(f'Hexagonal sampling {layout}')
        nside = 10
        pitch = 1.
        num_events = 100000
        grid = HexagonalGrid(layout, nside, nside, pitch)
        display = HexagonalGridDisplay(grid)
        display.draw(pixel_labels=True)
        plt.plot(0., 0., 'o', color='k')
        x = np.random.normal(0., 1., num_events)
        y = np.random.normal(0., 1., num_events)
        col, row = grid.world_to_pixel(x, y)
        mask = (col == target_col) * (row == target_row)
        plt.scatter(x[mask], y[mask], color='r', s=4.)
        mask = np.logical_not(mask)
        plt.scatter(x[mask], y[mask], color='b', s=4.)
        display.setup_gca()
    plt.show()
