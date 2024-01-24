# Copyright (C) 2024 luca.baldini@pi.infn.it
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

"""Initial brainstorming on xpol~4 design.
"""

from matplotlib.patches import Rectangle, Circle
import numpy as np

from hexsample import logger
from hexsample.hexagon import HexagonalLayout, HexagonalGrid
from hexsample.display import HexagonalGridDisplay
from hexsample.plot import plt
from hexsample.xpol import XPOL3_SIZE, XPOL_PITCH


def display_adc(num_cols: int = 9, num_rows: int = 6, pitch: float = 50.,
    layout=HexagonalLayout.ODD_Q):
    """Draw a sketch of the muGPD.
    """
    plt.figure(f'XPOL 4', figsize=(15., 9.))
    grid = HexagonalGrid(layout, num_cols, num_rows, pitch)
    display = HexagonalGridDisplay(grid)
    display.draw()
    for col in np.arange(num_cols):
        for row in np.arange(num_rows):
            x, y = grid.pixel_to_world(col, row)
            adc_index = (col % 3 + row * 3) % 9
            plt.text(x, y, f'{adc_index}', ha='center', va='center')
    plt.margins(0.01, 0.01)
    display.setup_gca()


if __name__ == '__main__':
    display_adc()
