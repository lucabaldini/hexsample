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


SAMPLE_PIXELS = ((12, 2), (9, 3), (12, 6), (9, 7), (13, 11), (9, 11), (13, 15))
BLOCK_COLOR = '#888'
RECT_COLOR = '#ddd'
EVENT_COLOR = '#bbb'


def display_adc_7(num_cols: int = 15, num_rows: int = 18, pitch: float = 50.,
    layout=HexagonalLayout.EVEN_R):
    """Draw a sketch of the muGPD.
    """
    magic_sequence = (0, 1, 5, 6, 2, 3, 4)
    start_adc = (0, 2, 5, 4, 2, 1, 4, 6, 1, 3, 6, 0, 3, 5)
    block_pixels = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1))
    plt.figure(f'XPOL 4 readout 7', figsize=(9., 9.))
    grid = HexagonalGrid(layout, num_cols, num_rows, pitch)
    display = HexagonalGridDisplay(grid)
    collection = display.draw()
    color = np.full((num_cols, num_rows), 'none')
    for i in range(7):
        for j in range(14):
            color[i, j] = RECT_COLOR
    for i, j in block_pixels:
        color[i, j] = BLOCK_COLOR
    for pixel in SAMPLE_PIXELS:
        color[pixel] = BLOCK_COLOR
        for i, j in grid.neighbors(*pixel):
            color[i, j] = EVENT_COLOR
    collection.set_facecolor(color.transpose().flatten())
    for col in np.arange(num_cols):
        for row in np.arange(num_rows):
            x, y = grid.pixel_to_world(col, row)
            start = magic_sequence.index(start_adc[row % len(start_adc)])
            index = (col + start) % len(magic_sequence)
            adc_index = magic_sequence[index]
            plt.text(x, y, f'{adc_index}', ha='center', va='center', size='small')
    plt.margins(0.01, 0.01)
    display.setup_gca()

def display_adc_9(num_cols: int = 15, num_rows: int = 18, pitch: float = 50.,
    layout=HexagonalLayout.EVEN_R):
    """Draw a sketch of the muGPD.
    """
    block_pixels = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2))
    plt.figure(f'XPOL 4 readout 9', figsize=(9., 9.))
    grid = HexagonalGrid(layout, num_cols, num_rows, pitch)
    display = HexagonalGridDisplay(grid)
    collection = display.draw()
    color = np.full((num_cols, num_rows), 'none')
    for i, j in block_pixels:
        color[i, j] = BLOCK_COLOR
    for pixel in SAMPLE_PIXELS:
        color[pixel] = EVENT_COLOR
        for i, j in grid.neighbors(*pixel):
            color[i, j] = EVENT_COLOR
    collection.set_facecolor(color.transpose().flatten())
    for col in np.arange(num_cols):
        for row in np.arange(num_rows):
            x, y = grid.pixel_to_world(col, row)
            adc_index = (col % 3 + row * 3) % 9
            plt.text(x, y, f'{adc_index}', ha='center', va='center', size='small')
    plt.margins(0.01, 0.01)
    display.setup_gca()

def display_template(num_cols: int = 16, num_rows: int = 12, pitch: float = 50.,
    layout=HexagonalLayout.ODD_Q):
    plt.figure(f'Hexagonal template', figsize=(15., 9.))
    grid = HexagonalGrid(layout, num_cols, num_rows, pitch)
    display = HexagonalGridDisplay(grid)
    display.draw()
    plt.margins(0.01, 0.01)
    display.setup_gca()



if __name__ == '__main__':
    display_adc_7()
    display_adc_9()
    plt.show()
    #display_template()
