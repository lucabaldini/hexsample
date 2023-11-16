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

"""Temporary code for brainstorming.
"""

from matplotlib.patches import Rectangle, Circle
import numpy as np

from hexsample import logger
from hexsample.hexagon import HexagonalLayout, HexagonalGrid
from hexsample.display import HexagonalGridDisplay
from hexsample.plot import plt
from hexsample.xpol import XPOL3_SIZE, XPOL_PITCH


def display_mugpd(centered : bool, num_cols : int = 10, num_rows : int = 6, pitch : float = 50.,
    layout=HexagonalLayout.ODD_Q, al_width : float = 6., oxide_width : float = 10.,
    al_thickness : float = 2., oxide_thickness : float = 2.):
    """Draw a sketch of the muGPD.
    """
    plt.figure(f'mugpd {layout} centered = {centered}', figsize=(15., 9.))
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
        r = Rectangle((x0 - 0.5 * oxide_width, ytop), oxide_width, strip_length, facecolor='white')
        plt.gca().add_patch(r)
        r = Rectangle((x0 - 0.5 * al_width, ytop), al_width, strip_length, facecolor='black')
        plt.gca().add_patch(r)
        # Draw the side view.
        r = Rectangle((x0 - 0.5 * oxide_width, yside), oxide_width, oxide_thickness, facecolor='white')
        plt.gca().add_patch(r)
        r = Rectangle((x0 - 0.5 * al_width, yside + oxide_thickness), al_width, al_thickness, facecolor='black')
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
            x1, y1 = x0, yside + oxide_thickness
            x2, y2 =_x, _y + scale * oxide_thickness
            r = Rectangle((_x - 0.5 * scale * oxide_width, _y), scale * oxide_width,
                scale * oxide_thickness, facecolor='white')
            plt.gca().add_patch(r)
            r = Rectangle((_x - 0.5 * scale * al_width, _y + scale * oxide_thickness),
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
        # Dimensioning: width of the oxide strip...
        if col == 0:
            x = (x0 - 0.5 * oxide_width, x0 + 0.5 * oxide_width)
            plt.vlines(x, ytop - 5., ytop - 50., color='gray', lw=1.)
            plt.text(x0, ytop - 60., f'{oxide_width:.0f} $\mu$m', ha='center', va='top')
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

def display_mugpd2d(num_cols : int = 10, num_rows : int = 6, pitch : float = 50.,
    layout=HexagonalLayout.ODD_Q, al_width : float = 6., oxide_width : float = 10.,
    al_thickness : float = 2., oxide_thickness : float = 2.):
    """Draw a sketch of the muGPD in the 2-dimensional falvor.
    """
    plt.figure(f'mugpd {layout} 2d', figsize=(9., 9.))
    grid = HexagonalGrid(layout, num_cols, num_rows, pitch)
    display = HexagonalGridDisplay(grid)
    display.draw()
    strip_length = 1.1 * grid.pitch * num_rows
    yside = 0.5 * strip_length + 0.5 * grid.pitch
    ytop = -0.5 * strip_length
    # First draw the vertical oxide...
    for col in np.arange(num_cols - 1):
        x0, _ = grid.pixel_to_world(col, 0)
        x0 += 0.5 * grid.secondary_pitch
        r = Rectangle((x0 - 0.5 * oxide_width, ytop), oxide_width, strip_length, facecolor='white')
        plt.gca().add_patch(r)
    # ... then the diagonal strips...
    xmin, ymax = grid.pixel_to_world(0, 0)
    xmax, ymin = grid.pixel_to_world(num_cols - 1, num_rows - 1)
    ymax += 0.5 * grid.pitch
    ymin -= 0.5 * grid.pitch
    theta = np.radians(30.)
    sin_ = np.sin(theta)
    cos_ = np.cos(theta)
    tan_ = np.tan(theta)
    delta_oxide = 0.5 * oxide_width / cos_
    delta_al = 0.5 * al_width / cos_
    for row in np.arange(num_rows + num_cols // 2 - 1):
        x1, y1 = grid.pixel_to_world(0, row)
        y1 -= 0.5 * grid.pitch
        if y1 < ymin:
            x1 -= (y1 - ymin) / tan_
            y1 = ymin
        y2 = ymax
        x2 = x1 + (y2 - y1) / tan_
        if x2 > xmax:
            y2 += (xmax - x2) * tan_
            x2 = xmax
        plt.fill_between((x1, x2), (y1 - delta_oxide, y2 - delta_oxide),
            (y1 + delta_oxide, y2 + delta_oxide), edgecolor='black', facecolor='white')
        plt.fill_between((x1, x2), (y1 - delta_al, y2 - delta_al),
            (y1 + delta_al, y2 + delta_al), color='black')
    # ... and, finally, the vertical aluminum---this will render the grid as
    # contiguous.
    for col in np.arange(num_cols - 1):
        x0, _ = grid.pixel_to_world(col, 0)
        x0 += 0.5 * grid.secondary_pitch
        r = Rectangle((x0 - 0.5 * al_width, ytop), al_width, strip_length, facecolor='black')
        plt.gca().add_patch(r)
    # And setup the plot.
    plt.margins(0.01, 0.01)
    display.setup_gca()

def test_exposed_dielectric():
    """All dimensions in cm.
    """
    pitch = XPOL_PITCH
    num_cols, num_rows = XPOL3_SIZE
    num_holes = num_cols * num_rows
    num_strips = num_rows
    strip_length = num_cols * pitch
    strip_thickness = 0.0002
    strip_al_width = 0.0006
    strip_oxide_width = 0.0010
    active_area = strip_length * num_rows * pitch * np.sqrt(3.) / 2.
    gem_thickness = 0.005
    gem_hole_diameter = 0.003
    gem_hole_surface = np.pi * gem_hole_diameter * gem_thickness * num_holes
    if True:
        # Add the GEM rims.
        r = gem_hole_diameter / 2.
        dr = 0.0002
        gem_hole_surface += 2. * np.pi * ((r + dr)**2. - r**2.) * num_holes
    mg_oxide_surface = strip_length * (strip_thickness + strip_oxide_width - strip_al_width) * num_strips
    logger.info(f'Number of pixels/GEM holes: {num_holes}')
    logger.info(f'Strip length: {strip_length} cm')
    logger.info(f'Active area: {active_area} cm^2')
    logger.info(f'GEM dielectric surface: {gem_hole_surface} cm^2 ({gem_hole_surface / active_area})')
    logger.info(f'muGPD dielectric surface: {mg_oxide_surface} cm^2 ({mg_oxide_surface / active_area})')

def draw_strip_group(x0, y0, num_strips, length, pitch, pad_pos=None, pad_side=0.025,
    bidimensional=False):
    """
    """
    fmt = dict(lw=0.75, color='black')
    x = (x0, x0 + num_strips * pitch)
    if y0 < 0:
        y = (y0, y0 + length)
    else:
        y = (y0, y0 - length)
    plt.vlines(np.linspace(*x, num_strips), *y, **fmt)
    if bidimensional:
        plt.hlines(np.linspace(*y, num_strips), *x, **fmt)
    if pad_pos is not None:
        xpad, ypad = pad_pos
        r = Rectangle(pad_pos, pad_side, pad_side, facecolor='black')
        plt.gca().add_patch(r)
        xor = min(x) if xpad < min(x) else max(x)
        yor = min(y) if ypad < min(y) else max(y)
        plt.hlines(yor, min(x), max(x), **fmt)
        if xpad > 0:
            xpad += pad_side
        if ypad > 0:
            ypad += pad_side
        plt.plot((xpad, xor), (ypad, yor), **fmt)

def test_gain_structures(strip_length=0.3, strip_padding=0.1, pad_padding=0.025):
    """
    """
    pitch = XPOL_PITCH
    strip_pitch = pitch * np.sqrt(3.) / 2.
    num_cols, num_rows = XPOL3_SIZE
    side = num_cols * pitch
    plt.figure('muGPD gain structures', figsize=(8., 8.))
    r = Rectangle((-side / 2., -side / 2.), side, side, facecolor='white')
    plt.gca().add_patch(r)
    label_fmt = dict(ha='center', va='center', backgroundcolor='white', size='x-small',
        bbox=dict(boxstyle='round,pad=0', fc='white', ec='none'))
    n = 64
    l = 0.9
    # 64 strips at nominal pitch, CENTERED and OFFSET configuration
    x0, y0 = -0.5 * side + strip_padding, -0.5 * side + strip_padding
    xpad, ypad = -0.5 * side + pad_padding, -0.5 * side + pad_padding
    draw_strip_group(x0, y0, n, l, strip_pitch, (xpad, ypad))
    plt.text(x0 + n // 2 * strip_pitch, y0 + 0.5 * l, f'64 @ 43.3 um\ncenter\n({n * l:.0f} pF)', **label_fmt)
    x0 = 0.5 * side - strip_padding - n * strip_pitch
    xpad = 0.5 * side - 2. * pad_padding
    draw_strip_group(x0, y0, n, l, strip_pitch, (xpad, ypad))
    plt.text(x0 + n // 2 * strip_pitch, y0 + 0.5 * l, f'64 @ 43.3 um\noffset\n({n * l:.0f} pF)', **label_fmt)
    # 32 strips at nominal pitch, CENTERED and OFFSET configuration
    n = 32
    l = side - 2. * strip_padding
    x0 = -0.2
    xpad = -0.5 * side + 2.5 * pad_padding
    draw_strip_group(x0, y0, n, l, strip_pitch, (xpad, ypad))
    plt.text(x0 + n // 2 * strip_pitch, y0 + 0.5 * l, f'32 @ 43.3 um\ncenter\n({n * l:.0f} pF)', **label_fmt)
    x0 = -x0 - n * strip_pitch
    xpad = 0.5 * side - 3.5 * pad_padding
    draw_strip_group(x0, y0, n, l, strip_pitch, (xpad, ypad))
    plt.text(x0 + n // 2 * strip_pitch, y0 + 0.5 * l, f'32 @ 43.3 um\noffset\n({n * l:.0f} pF)', **label_fmt)
    #
    n = 32
    l = 0.3
    x0, y0 = -0.5 * side + strip_padding, 0.5 * side - strip_padding
    xpad, ypad = -0.5 * side + pad_padding, 0.5 * side - 2. * pad_padding
    draw_strip_group(x0, y0, n, l, 2. * strip_pitch, (xpad, ypad))
    plt.text(x0 + n * strip_pitch, y0 - 0.5 * l, f'32 @ 86.6 um\n({n * l:.0f} pF)', **label_fmt)
    #
    n = 64
    l = 0.3
    x0, y0 = 0.5 * side - strip_padding - n * strip_pitch, 0.5 * side - strip_padding
    xpad, ypad = 0.5 * side - 2. * pad_padding, 0.5 * side - 2. * pad_padding
    draw_strip_group(x0, y0, n, l, strip_pitch, (xpad, ypad), bidimensional=True)
    plt.text(x0 + n // 2 * strip_pitch, y0 - 0.5 * l, f'2d structure', **label_fmt)
    # Setup the plot.
    plt.gca().set_aspect('equal')
    plt.gca().autoscale()
    plt.axis('off')

def display_process(num_strips=3, al_width=6., oxide_width=10.):
    """Sketch the process.
    """
    pitch = XPOL_PITCH
    strip_pitch = 10000. * pitch * np.sqrt(3.) / 2.
    side = num_strips * strip_pitch
    bulk_thickness = 20.
    chip_top_thickness = 1.
    oxide_thickness = 3.
    al_thickness = 3.
    text_fmt = dict(va='center', size='small')
    plt.figure('muGPD process', figsize=(11., 2.))
    plt.gca().set_aspect('equal')
    plt.axis('off')
    bulk = Rectangle((0., -bulk_thickness), side, bulk_thickness, facecolor='white', hatch='//')
    plt.gca().add_patch(bulk)
    top_layer = Rectangle((0., 0.), side, chip_top_thickness, facecolor='black')
    plt.gca().add_patch(top_layer)
    y = 0.5 * chip_top_thickness
    plt.text(num_strips * strip_pitch, y, ' ASIC top layer', ha='left', **text_fmt)
    plt.gca().autoscale()
    plt.savefig('mugp_process_1.pdf')
    oxide = Rectangle((0., chip_top_thickness), side, oxide_thickness, facecolor='white')
    plt.gca().add_patch(oxide)
    y = chip_top_thickness + 0.5 * oxide_thickness
    plt.text(0., y, 'Oxide ', ha='right', **text_fmt)
    plt.savefig('mugp_process_2.pdf')
    y = chip_top_thickness + oxide_thickness
    strip = Rectangle((0., y), side, al_thickness, facecolor='black')
    plt.gca().add_patch(strip)
    y = chip_top_thickness + oxide_thickness + 0.5 * al_thickness
    plt.text(0., y, 'Metal ', ha='right', **text_fmt)
    plt.savefig('mugp_process_3.pdf')
    strip.remove()
    y = chip_top_thickness + oxide_thickness
    for i in range(num_strips):
        s = Rectangle(((i + 0.5) * strip_pitch, y), al_width, al_thickness, facecolor='black')
        plt.gca().add_patch(s)
    plt.savefig('mugp_process_4.pdf')
    oxide.remove()
    y -= al_thickness
    for i in range(num_strips):
        s = Rectangle(((i + 0.5) * strip_pitch - 0.5 * (oxide_width - al_width), y), oxide_width, oxide_thickness, facecolor='white')
        plt.gca().add_patch(s)
    plt.savefig('mugp_process_5.pdf')

def display_strip_design():
    """
    """
    plt.figure('Strip design', figsize=(8., 5.))
    pitch = XPOL_PITCH
    strip_pitch = 10000. * pitch * np.sqrt(3.) / 2.
    chip_top_thickness = 0.5
    oxide_thickness = 2.
    oxide_width = 10.
    al_thickness = 2.
    al_width = 6.
    side = 20.
    top_layer = Rectangle((0., 0.), side, chip_top_thickness, facecolor='black')
    plt.gca().add_patch(top_layer)
    x, y = 0.5 * (side - oxide_width), chip_top_thickness
    oxide = Rectangle((x, y), oxide_width, oxide_thickness, facecolor='white')
    plt.gca().add_patch(oxide)
    plt.text(x + 0.5 * oxide_width, y + 0.5 * oxide_thickness, 'Oxide', ha='center', va='center')
    x, y = 0.5 * (side - al_width), chip_top_thickness + oxide_thickness
    al = Rectangle((x, y), al_width, al_thickness, facecolor='black')
    plt.gca().add_patch(al)
    plt.text(x + 0.5 * al_width, y + 0.5 * al_thickness, 'Al', color='white', ha='center', va='center')
    ytop = y + al_thickness
    plt.vlines((x, x + al_width), ytop + 0.5, ytop + 3., color='gray', lw=1.)
    plt.text(x + 0.5 * al_width, ytop + 3., f'{al_width:.1f} $\mu$m', ha='center', va='top')
    plt.hlines((y, ytop), x + 0.85 * oxide_width, 20, color='gray', lw=1.)
    plt.text(18., y + 0.5 * al_thickness, f'{al_thickness:.1f} $\mu$m', ha='left', va='center')
    plt.text(18., y - 0.5 * oxide_thickness, f'{oxide_thickness:.1f} $\mu$m', ha='left', va='center')
    offset = -10.
    oxide_width = al_width
    oxide_thickness = 3.
    top_layer = Rectangle((0., offset), side, chip_top_thickness, facecolor='black')
    plt.gca().add_patch(top_layer)
    x, y = 0.5 * (side - oxide_width), offset + chip_top_thickness
    oxide = Rectangle((x, y), oxide_width, oxide_thickness, facecolor='white')
    plt.gca().add_patch(oxide)
    plt.text(x + 0.5 * oxide_width, y + 0.5 * oxide_thickness, 'Oxide', ha='center', va='center')
    x, y = 0.5 * (side - al_width), offset + chip_top_thickness + oxide_thickness
    al = Rectangle((x, y), al_width, al_thickness, facecolor='black')
    plt.gca().add_patch(al)
    plt.text(x + 0.5 * al_width, y + 0.5 * al_thickness, 'Al', color='white', ha='center', va='center')
    plt.hlines((y, y + al_thickness), x + 1.1 * oxide_width, 20, color='gray', lw=1.)
    plt.text(18., y + 0.5 * al_thickness, f'{al_thickness:.1f} $\mu$m', ha='left', va='center')
    plt.text(18., y - 0.5 * oxide_thickness, f'{oxide_thickness:.1f} $\mu$m', ha='left', va='center')
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.gca().autoscale()


def test_display():
    """Display the two possible simplest arrangements.
    """
    test_gain_structures()
    display_mugpd(True)
    display_mugpd(False)
    display_mugpd2d()
    test_exposed_dielectric()
    display_process()
    display_strip_design()



if __name__ == '__main__':
    test_display()
    plt.show()
