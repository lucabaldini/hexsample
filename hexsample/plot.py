# Copyright (C) 2023 the baldaquin team.
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


"""matplotlib configuration module.
"""

from pathlib import Path
import sys
from typing import Any

from loguru import logger
import matplotlib
from matplotlib import pyplot as plt

if sys.flags.interactive:
    plt.ion()


DEFAULT_FIG_WIDTH = 8.

DEFAULT_FIG_HEIGHT = 6.

DEFAULT_FIG_SIZE = (DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT)

DEFAULT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]


SERIF_FONTS = [
    'DejaVu Serif', 'Bitstream Vera Serif', 'New Century Schoolbook',
    'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman',
    'Nimbus Roman No9 L', 'Times New Roman', 'Times', 'Palatino', 'Charter',
    'serif'
]

SANS_SERIF_FONTS = [
    'DejaVu Sans', 'Bitstream Vera Sans', 'Lucida Grande', 'Verdana', 'Geneva',
    'Lucid', 'Arial', 'Helvetica', 'Avant Garde', 'sans-serif'
]

CURSIVE_FONTS = [
    'Apple Chancery', 'Textile', 'Zapf Chancery', 'Sand', 'Script MT',
    'Felipa', 'cursive'
]

FANTASY_FONTS = [
    'Comic Sans MS', 'Chicago', 'Charcoal', 'Impact', 'Western', 'Humor Sans',
    'xkcd', 'fantasy'
]

MONOSPACE_FONTS = [
    'DejaVu Sans Mono', 'Bitstream Vera Sans Mono', 'Andale Mono',
    'Nimbus Mono L', 'Courier New', 'Courier', 'Fixed', 'Terminal', 'monospace'
]



class PlotCard(dict):

    """Small class reperesenting a text card.

    This is essentially a dictionary that is capable of plotting itself on
    a matplotlib figure in the form of a multi-line graphic card.

    Arguments
    ---------
    data : dict
        A dictionary holding the lines to be displayed in the card.
    """

    KEY_KWARGS = dict(color='gray', size='x-small', ha='left', va='top')
    VALUE_KWARGS = dict(color='black', size='small', ha='left', va='top')

    def __init__(self, data : dict = None) -> None:
        """Constructor.
        """
        super().__init__()
        if data is not None:
            for key, value in data.items():
                self.add_line(key, value)

    def add_line(self, key : str, value : float, fmt : str = '%g', units : str = None) -> None:
        """Set the value for a given key.

        Arguments
        ---------
        key : str
            The key, i.e., the explanatory text for a given value.

        value : float, optional
            The actual value (if None, a blank line will be added).

        fmt : str
            The string format to be used to render the value.

        units : str
            The measurement units for the value.
        """
        self[key] = (value, fmt, units)

    def draw(self, axes = None, x : float = 0.05, y : float = 0.95, line_spacing : float = 0.075,
        spacing_ratio : float = 0.75) -> None:
        """Draw the card.

        Arguments
        ---------
        x0, y0 : float
            The absolute coordinates of the top-left corner of the card.

        line_spacing : float
            The line spacing in units of the total height of the current axes.

        spacing_ratio : float
            The fractional line spacing assigned to the key label.
        """
        # pylint: disable=invalid-name
        if axes is None:
            axes = plt.gca()
        key_norm = spacing_ratio / (1. + spacing_ratio)
        value_norm = 1. - key_norm
        for kwargs in (self.KEY_KWARGS, self.VALUE_KWARGS):
            kwargs['transform'] = axes.transAxes
        for key, (value, fmt, units) in self.items():
            if value is None:
                y -= 0.5 * line_spacing
                continue
            axes.text(x, y, key, **self.KEY_KWARGS)
            y -= key_norm * line_spacing
            value = fmt % value
            if units is not None:
                value = f'{value} {units}'
            axes.text(x, y, value, **self.VALUE_KWARGS)
            y -= value_norm * line_spacing



def last_line_color(default : str = 'black') -> str:
    """Return the color used to draw the last line
    """
    try:
        return plt.gca().get_lines()[-1].get_color()
    except IndexError:
        return default

def setup_axes(axes, **kwargs) -> None:
    """Setup a generic axes object.
    """
    if kwargs.get('logx'):
        axes.set_xscale('log')
    if kwargs.get('logy'):
        axes.set_yscale('log')
    xticks = kwargs.get('xticks')
    if xticks is not None:
        axes.set_xticks(xticks)
    yticks = kwargs.get('yticks')
    if yticks is not None:
        axes.set_yticks(yticks)
    xlabel = kwargs.get('xlabel')
    if xlabel is not None:
        axes.set_xlabel(xlabel)
    ylabel = kwargs.get('ylabel')
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    xmin, xmax, ymin, ymax = [kwargs.get(key) for key in ('xmin', 'xmax', 'ymin', 'ymax')]
    if xmin is None and xmax is None and ymin is None and ymax is None:
        pass
    else:
        axes.axis([xmin, xmax, ymin, ymax])
    if kwargs.get('grids'):
        axes.grid(which='both')
    if kwargs.get('legend'):
        axes.legend()

def setup_gca(**kwargs) -> None:
    """Setup the axes for the current plot.
    """
    setup_axes(plt.gca(), **kwargs)

def save_gcf(folder_path : Path, file_extensions=('png', 'pdf'), **kwargs) -> None:
    """Save the current matplotlib figure.
    """
    figure_name = plt.gcf().get_label().lower().replace(' ', '_')
    for extension in file_extensions:
        file_path = folder_path / f'{figure_name}.{extension}'
        logger.info(f'Saving current figure to {file_path}...')
        plt.savefig(file_path, **kwargs)

def _set_rc_param(key : str, value : Any):
    """Set the value for a single matplotlib parameter.

    The actual command is encapsulated into a try except block because this
    is intended to work across different matplotlib versions. If a setting
    cannot be applied for whatever reason, this will happily move on.
    """
    try:
        matplotlib.rcParams[key] = value
    except KeyError:
        logger.warning(f'Unknown matplotlib rc param {key}, skipping...')
    except ValueError as exception:
        logger.warning(f'{exception}, skipping...')

def setup():
    """Basic system-wide setup.

    The vast majority of the settings are taken verbatim from the
    matplotlib 2, commit 5285e76:
    https://github.com/matplotlib/matplotlib/blob/master/matplotlibrc.template

    Note that, although this is designed to provide an experience which is as
    consistent as possible across different matplotlib versions, some of the
    functionalities are not implemented in older versions, which is why we wrap
    each parameter setting into a _set_rc_param() function call.
    """
    # pylint: disable=too-many-statements
    # http://matplotlib.org/api/artist_api.html#module-matplotlib.lines
    _set_rc_param('lines.linewidth', 1.5)
    _set_rc_param('lines.linestyle', '-')
    _set_rc_param('lines.color', DEFAULT_COLORS[0])
    _set_rc_param('lines.marker', None)
    _set_rc_param('lines.markeredgewidth', 1.0)
    _set_rc_param('lines.markersize', 6)
    _set_rc_param('lines.dash_joinstyle', 'miter')
    _set_rc_param('lines.dash_capstyle', 'butt')
    _set_rc_param('lines.solid_joinstyle', 'miter')
    _set_rc_param('lines.solid_capstyle', 'projecting')
    _set_rc_param('lines.antialiased', True)
    _set_rc_param('lines.dashed_pattern', (2.8, 1.2))
    _set_rc_param('lines.dashdot_pattern', (4.8, 1.2, 0.8, 1.2))
    _set_rc_param('lines.dotted_pattern', (1.1, 1.1))
    _set_rc_param('lines.scale_dashes', True)

    # Markers.
    _set_rc_param('markers.fillstyle', 'full')

    # http://matplotlib.org/api/artist_api.html#module-matplotlib.patches
    _set_rc_param('patch.linewidth', 1)
    _set_rc_param('patch.facecolor', DEFAULT_COLORS[0])
    _set_rc_param('patch.edgecolor', 'black')
    _set_rc_param('patch.force_edgecolor', True)
    _set_rc_param('patch.antialiased', True)

    # Hatches
    _set_rc_param('hatch.color', 'k')
    _set_rc_param('hatch.linewidth', 1.0)

    # Boxplot
    _set_rc_param('boxplot.notch', False)
    _set_rc_param('boxplot.vertical', True)
    _set_rc_param('boxplot.whiskers', 1.5)
    _set_rc_param('boxplot.bootstrap', None)
    _set_rc_param('boxplot.patchartist', False)
    _set_rc_param('boxplot.showmeans', False)
    _set_rc_param('boxplot.showcaps', True)
    _set_rc_param('boxplot.showbox', True)
    _set_rc_param('boxplot.showfliers', True)
    _set_rc_param('boxplot.meanline', False)
    _set_rc_param('boxplot.flierprops.color', 'k')
    _set_rc_param('boxplot.flierprops.marker', 'o')
    _set_rc_param('boxplot.flierprops.markerfacecolor', 'none')
    _set_rc_param('boxplot.flierprops.markeredgecolor', 'k')
    _set_rc_param('boxplot.flierprops.markersize', 6)
    _set_rc_param('boxplot.flierprops.linestyle', 'none')
    _set_rc_param('boxplot.flierprops.linewidth', 1.0)
    _set_rc_param('boxplot.boxprops.color', 'k')
    _set_rc_param('boxplot.boxprops.linewidth', 1.0)
    _set_rc_param('boxplot.boxprops.linestyle', '-')
    _set_rc_param('boxplot.whiskerprops.color', 'k')
    _set_rc_param('boxplot.whiskerprops.linewidth', 1.0)
    _set_rc_param('boxplot.whiskerprops.linestyle', '-')
    _set_rc_param('boxplot.capprops.color', 'k')
    _set_rc_param('boxplot.capprops.linewidth', 1.0)
    _set_rc_param('boxplot.capprops.linestyle', '-')
    _set_rc_param('boxplot.medianprops.color', DEFAULT_COLORS[1])
    _set_rc_param('boxplot.medianprops.linewidth', 1.0)
    _set_rc_param('boxplot.medianprops.linestyle', '-')
    _set_rc_param('boxplot.meanprops.color', DEFAULT_COLORS[2])
    _set_rc_param('boxplot.meanprops.marker', '^')
    _set_rc_param('boxplot.meanprops.markerfacecolor', DEFAULT_COLORS[2])
    _set_rc_param('boxplot.meanprops.markeredgecolor', DEFAULT_COLORS[2])
    _set_rc_param('boxplot.meanprops.markersize', 6)
    _set_rc_param('boxplot.meanprops.linestyle', 'none')
    _set_rc_param('boxplot.meanprops.linewidth', 1.0)

    # http://matplotlib.org/api/font_manager_api.html for more
    _set_rc_param('font.family', 'sans-serif')
    _set_rc_param('font.style', 'normal')
    _set_rc_param('font.variant', 'normal')
    _set_rc_param('font.weight', 'medium')
    _set_rc_param('font.stretch', 'normal')
    _set_rc_param('font.size', 14.0)
    _set_rc_param('font.serif', SERIF_FONTS)
    _set_rc_param('font.sans-serif', SANS_SERIF_FONTS)
    _set_rc_param('font.cursive', CURSIVE_FONTS)
    _set_rc_param('font.fantasy', FANTASY_FONTS)
    _set_rc_param('font.monospace', MONOSPACE_FONTS)

    # http://matplotlib.org/api/artist_api.html#module-matplotlib.text for more
    _set_rc_param('text.color', 'black')

    #http://wiki.scipy.org/Cookbook/Matplotlib/UsingTex
    _set_rc_param('text.usetex', False)
    _set_rc_param('text.hinting', 'auto')
    _set_rc_param('text.hinting_factor', 8)
    _set_rc_param('text.antialiased', True)
    _set_rc_param('mathtext.cal', 'cursive')
    _set_rc_param('mathtext.rm', 'serif')
    _set_rc_param('mathtext.tt', 'monospace')
    _set_rc_param('mathtext.it', 'serif:italic')
    _set_rc_param('mathtext.bf', 'serif:bold')
    _set_rc_param('mathtext.sf', 'sans')
    _set_rc_param('mathtext.fontset', 'stixsans')
    _set_rc_param('mathtext.default', 'it')

    # http://matplotlib.org/api/axes_api.html#module-matplotlib.axes
    _set_rc_param('axes.facecolor', 'white')
    _set_rc_param('axes.edgecolor', 'black')
    _set_rc_param('axes.linewidth', 1.25)
    _set_rc_param('axes.grid', False)
    _set_rc_param('axes.titlesize', 'large')
    _set_rc_param('axes.titlepad', 6.0)
    _set_rc_param('axes.labelsize', 'medium')
    _set_rc_param('axes.labelpad', 4.0)
    _set_rc_param('axes.labelweight', 'normal')
    _set_rc_param('axes.labelcolor', 'black')
    _set_rc_param('axes.axisbelow', 'line')
    _set_rc_param('axes.formatter.limits', (-7, 7))
    _set_rc_param('axes.formatter.use_locale', False)
    _set_rc_param('axes.formatter.use_mathtext', False)
    _set_rc_param('axes.formatter.min_exponent', 0)
    _set_rc_param('axes.formatter.useoffset', True)
    _set_rc_param('axes.formatter.offset_threshold', 4)
    _set_rc_param('axes.unicode_minus', True)
    _set_rc_param('axes.autolimit_mode', 'round_numbers')
    _set_rc_param('axes.xmargin', 0.)
    _set_rc_param('axes.ymargin', 0.)
    _set_rc_param('polaraxes.grid', True)
    _set_rc_param('axes3d.grid', True)

    # Dates
    _set_rc_param('date.autoformatter.year', '%Y')
    _set_rc_param('date.autoformatter.month', '%Y-%m')
    _set_rc_param('date.autoformatter.day', '%Y-%m-%d')
    _set_rc_param('date.autoformatter.hour', ' %m-%d %H')
    _set_rc_param('date.autoformatter.minute', '%d %H:%M')
    _set_rc_param('date.autoformatter.second', '%H:%M:%S')
    _set_rc_param('date.autoformatter.microsecond', '%M:%S.%f')

    # see http://matplotlib.org/api/axis_api.html#matplotlib.axis.Tick
    _set_rc_param('xtick.top', False)
    _set_rc_param('xtick.bottom', True)
    _set_rc_param('xtick.major.size', 3.5)
    _set_rc_param('xtick.minor.size', 2)
    _set_rc_param('xtick.major.width', 1.25)
    _set_rc_param('xtick.minor.width', 1.)
    _set_rc_param('xtick.major.pad', 3.5)
    _set_rc_param('xtick.minor.pad', 3.4)
    _set_rc_param('xtick.color', 'k')
    _set_rc_param('xtick.labelsize', 'medium')
    _set_rc_param('xtick.direction', 'out')
    _set_rc_param('xtick.minor.visible', False)
    _set_rc_param('xtick.major.top', True)
    _set_rc_param('xtick.major.bottom', True)
    _set_rc_param('xtick.minor.top', True)
    _set_rc_param('xtick.minor.bottom', True)
    _set_rc_param('ytick.left', True)
    _set_rc_param('ytick.right', False)
    _set_rc_param('ytick.major.size', 3.5)
    _set_rc_param('ytick.minor.size', 2)
    _set_rc_param('ytick.major.width', 1.25)
    _set_rc_param('ytick.minor.width', 1.)
    _set_rc_param('ytick.major.pad', 3.5)
    _set_rc_param('ytick.minor.pad', 3.4)
    _set_rc_param('ytick.color', 'k')
    _set_rc_param('ytick.labelsize', 'medium')
    _set_rc_param('ytick.direction', 'out')
    _set_rc_param('ytick.minor.visible', False)
    _set_rc_param('ytick.major.left', True)
    _set_rc_param('ytick.major.right', True)
    _set_rc_param('ytick.minor.left', True)
    _set_rc_param('ytick.minor.right', True)

    # Grids
    _set_rc_param('grid.color', '#c0c0c0')
    _set_rc_param('grid.linestyle', '--')
    _set_rc_param('grid.linewidth', 0.8)
    _set_rc_param('grid.alpha', 1.0)

    # Legend
    _set_rc_param('legend.loc', 'best')
    _set_rc_param('legend.frameon', True)
    _set_rc_param('legend.framealpha', 0.8)
    _set_rc_param('legend.facecolor', 'inherit')
    _set_rc_param('legend.edgecolor', 'gray')
    _set_rc_param('legend.fancybox', True)
    _set_rc_param('legend.shadow', False)
    _set_rc_param('legend.numpoints', 1)
    _set_rc_param('legend.scatterpoints', 1)
    _set_rc_param('legend.markerscale', 1.0)
    _set_rc_param('legend.fontsize', 'medium')
    _set_rc_param('legend.borderpad', 0.4)
    _set_rc_param('legend.labelspacing', 0.5)
    _set_rc_param('legend.handlelength', 2.0)
    _set_rc_param('legend.handleheight', 0.7)
    _set_rc_param('legend.handletextpad', 0.8)
    _set_rc_param('legend.borderaxespad', 0.5)
    _set_rc_param('legend.columnspacing', 2.0)

    # See http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure
    _set_rc_param('figure.titlesize', 'large')
    _set_rc_param('figure.titleweight', 'normal')
    _set_rc_param('figure.figsize', DEFAULT_FIG_SIZE)
    _set_rc_param('figure.dpi', 100)
    _set_rc_param('figure.facecolor', 'white')
    _set_rc_param('figure.edgecolor', 'white')
    _set_rc_param('figure.autolayout', False)
    _set_rc_param('figure.max_open_warning', 20)
    _set_rc_param('figure.subplot.left', 0.125)
    _set_rc_param('figure.subplot.right', 0.95)
    _set_rc_param('figure.subplot.bottom', 0.10)
    _set_rc_param('figure.subplot.top', 0.95)
    _set_rc_param('figure.subplot.wspace', 0.2)
    _set_rc_param('figure.subplot.hspace', 0.2)

    # Images
    _set_rc_param('image.aspect', 'equal')
    _set_rc_param('image.interpolation', 'nearest')
    _set_rc_param('image.cmap', 'jet')
    _set_rc_param('image.lut', 256)
    _set_rc_param('image.origin', 'upper')
    _set_rc_param('image.resample', True)
    _set_rc_param('image.composite_image', True)

    # Contour plots
    _set_rc_param('contour.negative_linestyle', 'dashed')
    _set_rc_param('contour.corner_mask', True)

    # Errorbar plots
    _set_rc_param('errorbar.capsize', 0)

    # Histogram plots
    _set_rc_param('hist.bins', 10)

    # Scatter plots
    _set_rc_param('scatter.marker', 'o')

    # Saving figures
    _set_rc_param('path.simplify', True)
    _set_rc_param('path.simplify_threshold', 0.1)
    _set_rc_param('path.snap', True)
    _set_rc_param('path.sketch', None)
    _set_rc_param('savefig.dpi', 'figure')
    _set_rc_param('savefig.facecolor', 'white')
    _set_rc_param('savefig.edgecolor', 'white')
    _set_rc_param('savefig.format', 'png')
    _set_rc_param('savefig.bbox', 'standard')
    _set_rc_param('savefig.pad_inches', 0.1)
    _set_rc_param('savefig.directory', '~')
    _set_rc_param('savefig.transparent', False)

    # Back-ends
    _set_rc_param('pdf.compression', 6)
    _set_rc_param('pdf.fonttype', 3)
    _set_rc_param('svg.image_inline', True)
    _set_rc_param('svg.fonttype', 'path')
    _set_rc_param('svg.hashsalt', None)

    # Key maps
    _set_rc_param('keymap.fullscreen', ('f', 'ctrl+f'))
    _set_rc_param('keymap.home', ('h', 'r', 'home'))
    _set_rc_param('keymap.back', ('left', 'c', 'backspace'))
    _set_rc_param('keymap.forward', ('right', 'v'))
    _set_rc_param('keymap.pan', 'p')
    _set_rc_param('keymap.zoom', 'o')
    _set_rc_param('keymap.save', 's')
    _set_rc_param('keymap.quit', ('ctrl+w', 'cmd+w'))
    _set_rc_param('keymap.grid', 'g')
    _set_rc_param('keymap.grid_minor', 'G')
    _set_rc_param('keymap.yscale', 'l')
    _set_rc_param('keymap.xscale', ('L', 'k'))

    # Animations settings
    _set_rc_param('animation.html', 'none')
    _set_rc_param('animation.writer', 'ffmpeg')
    _set_rc_param('animation.codec', 'h264')
    _set_rc_param('animation.bitrate', -1)
    _set_rc_param('animation.frame_format', 'png')
    _set_rc_param('animation.ffmpeg_path', 'ffmpeg')
    _set_rc_param('animation.ffmpeg_args', '')
    _set_rc_param('animation.convert_path', 'convert')

    # Miscellanea
    _set_rc_param('timezone', 'UTC')

setup()
