# Copyright (C) 2023--2025 the hexsample team.
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

"""Test suite for mc.py
"""

import pytest
from aptapy.plotting import plt

from hexsample import rng
from hexsample.display import HexagonalGridDisplay
from hexsample.hexagon import HexagonalGrid, HexagonalLayout
from hexsample.mc import MonteCarloEvent
from hexsample.readout import HexagonalReadoutRectangular
from hexsample.roi import Padding


@pytest.mark.skip(reason="intermittent failure, see issue #43")
def test_diffusion(diff_sigma=40.):
    """Test the diffusion.
    """
    rng.initialize()
    grid = HexagonalGrid(HexagonalLayout.ODD_R, 2, 2, 0.005)
    evt = MonteCarloEvent(0., 8000., 0., 0., 0.05, 3000)
    x, y = evt.propagate(diff_sigma)
    padding = Padding(1, 1, 1, 1)
    args = HexagonalLayout.ODD_R, 10, 10, 0.005, 40., 1., 500, 80, 0, padding
    readout = HexagonalReadoutRectangular(*args)
    digi_event = readout.read(evt.timestamp, x, y)
    print(digi_event.ascii())
    display = HexagonalGridDisplay(grid)
    display.draw()
    plt.plot(x, y, "o", markersize=1)
