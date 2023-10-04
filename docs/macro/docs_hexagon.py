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


from hexsample import HEXSAMPLE_DOCS_FIGURES
from hexsample.display import HexagonalGridDisplay
from hexsample.hexagon import HexagonalLayout, HexagonalGrid
from hexsample.plot import plt, save_gcf


def draw_hexagonal_layouts(nside : int = 4, pitch : float = 1.):
    """Draw all the possible hexagonal layouts for the docs..
    """
    plt.figure('Hexagonal layouts')
    dist = 1.5 * nside * pitch
    for i, layout in enumerate(HexagonalLayout):
        x, y = dist * (i % 2), -dist * (i // 2)
        grid = HexagonalGrid(layout, nside, nside, pitch)
        display = HexagonalGridDisplay(grid)
        display.draw(offset=(x, y), pixel_labels=True)
        plt.plot(x, y, 'o', color='k')
        plt.text(x, y + 0.52 * dist, layout, ha='center', size='small')
        axis_length = 0.42 * dist
        plt.plot((x, x + axis_length), (y, y), color='k')
        plt.text(x + axis_length, y, 'x', va='center', size='small')
        plt.plot((x, x), (y, y + axis_length), color='k')
        plt.text(x, y + axis_length, 'y', ha='center', va='bottom', size='small')
    display.setup_gca()
    plt.tight_layout()
    save_gcf(HEXSAMPLE_DOCS_FIGURES)
    plt.show()



if __name__ == '__main__':
    draw_hexagonal_layouts()
