# Copyright (C) 2023 luca.baldini@pi.infn.it
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


from hexsample.display import HexagonalGridDisplay
from hexsample.hexagon import HexagonalGrid, HexagonalLayout
from hexsample.mc import MonteCarloEvent
from hexsample.plot import plt
from hexsample.readout import HexagonalReadoutRectangular
from hexsample.roi import Padding


def test_diffusion(diff_sigma=40.):
    """
    """
    grid = HexagonalGrid(HexagonalLayout.ODD_R, 2, 2, 0.005)
    evt = MonteCarloEvent(0., 8000., 0., 0., 0.05, 3000)
    x, y = evt.propagate(diff_sigma)
    readout = HexagonalReadoutRectangular(HexagonalLayout.ODD_R, 10, 10, 0.005, 40., 1.)
    padding = Padding(1)
    digi_event = readout.read(evt.timestamp, x, y, 500., padding, 80, 0)
    print(digi_event.ascii())
    display = HexagonalGridDisplay(grid)
    display.draw()
    plt.plot(x, y, 'o', markersize=1)
    return display


if __name__ == '__main__':
    display = test_diffusion()
    display.show()
