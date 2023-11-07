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

"""Temporary code for brainstorming.
"""

from matplotlib.patches import Rectangle, Circle
import numpy as np

from hexsample.hexagon import HexagonalLayout, HexagonalGrid
from hexsample.display import HexagonalGridDisplay
from hexsample.plot import plt


def display_mggpd(centered : bool, num_cols : int = 10, num_rows : int = 6, pitch : float = 50.,
    layout=HexagonalLayout.ODD_Q, al_width : float = 6., oxyde_width : float = 10.,
    al_thickness : float = 2., oxyde_thickness : float = 2.):
    """Draw a sketch of the MGGPD.
    """
    plt.figure(f'MGGPD {layout} centered = {centered}', figsize=(15., 9.))
    grid = HexagonalGrid(layout, num_cols, num_rows, pitch)
    display = HexagonalGridDisplay(grid)
    display.draw()
    strip_length = 1.1 * grid.pitch * num_rows
    yside = 0.5 * strip_length + 0.5 * grid.pitch
    ytop = -0.5 * strip_length
    for col in np.arange(num_cols - 1 + centered):
        x0, _ = grid.pixel_to_world(col, 0)
        if not centered:
            x0 += 0.5 * grid.secondary_pitch
        # Draw the top view.
        r = Rectangle((x0 - 0.5 * oxyde_width, ytop), oxyde_width, strip_length, facecolor='white')
        plt.gca().add_patch(r)
        r = Rectangle((x0 - 0.5 * al_width, ytop), al_width, strip_length, facecolor='black')
        plt.gca().add_patch(r)
        # Draw the side view.
        r = Rectangle((x0 - 0.5 * oxyde_width, yside), oxyde_width, oxyde_thickness, facecolor='white')
        plt.gca().add_patch(r)
        r = Rectangle((x0 - 0.5 * al_width, yside + oxyde_thickness), al_width, al_thickness, facecolor='black')
        plt.gca().add_patch(r)
        # Complete the side view.
        if col == 0:
            dx = 0.5 * num_cols * grid.secondary_pitch
            plt.hlines(yside, -dx, dx, color='black')
            plt.text(-1.2 * dx, yside, 'Side view', ha='right', va='center')
            # Inset for the zoomed side view.
            radius = 10.
            scale = 5.
            _x, _y = -1.5 * dx, 0.5 * yside
            x1, y1 = x0, yside + oxyde_thickness
            x2, y2 =_x, _y + scale * oxyde_thickness
            r = Rectangle((_x - 0.5 * scale * oxyde_width, _y), scale * oxyde_width,
                scale * oxyde_thickness, facecolor='white')
            plt.gca().add_patch(r)
            r = Rectangle((_x - 0.5 * scale * al_width, _y + scale * oxyde_thickness),
                scale * al_width, scale * al_thickness, facecolor='black')
            plt.gca().add_patch(r)
            # Small circle.
            c = Circle(((x1, y1)), radius, fill=False)
            plt.gca().add_patch(c)
            # Large circle.
            c = Circle((x2, y2), scale * radius, fill=False)
            plt.gca().add_patch(c)
            # Connector.
            theta = np.arctan2(x2 - x1, y2 - y1)
            s = np.sin(theta)
            c = np.cos(theta)
            plt.plot((x1 + radius * s, x2 - scale * radius * s), (y1 + radius * c, y2 - scale * radius * c), color='black', lw=1.25)
        # Dimensioning: width of the oxyde strip...
        if col == 0:
            x = (x0 - 0.5 * oxyde_width, x0 + 0.5 * oxyde_width)
            plt.vlines(x, ytop - 5., ytop - 50., color='gray', lw=1.)
            plt.text(x0, ytop - 60., f'{oxyde_width:.0f} $\mu$m', ha='center', va='top')
            plt.text(-1.2 * dx, 0., 'Top view', ha='right', va='center')
        # ... width of the Al strip...
        if col == 1:
            x = (x0 - 0.5 * al_width, x0 + 0.5 * al_width)
            plt.vlines(x, ytop - 5., ytop - 50., color='gray', lw=1.)
            plt.text(x0, ytop - 60., f'{al_width:.0f} $\mu$m', ha='center', va='top')
        # ... secondary pitch of the grid.
        if col == num_cols - 2 + centered:
            p = grid.secondary_pitch
            x = (x0, x0 - p)
            plt.vlines(x, ytop - 5., ytop - 50., color='gray', lw=1.)
            plt.text(x0 - 0.5 * p, ytop - 60., f'{p:.2f} $\mu$m', ha='center', va='top')
    plt.margins(0.01, 0.01)
    display.setup_gca()

def test_display():
    """Display the two possible simplest arrangements.
    """
    display_mggpd(True)
    display_mggpd(False)



if __name__ == '__main__':
    test_display()
    plt.show()
