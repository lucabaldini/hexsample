# Copyright (C) 2023 luca.baldini@pi.infn.it, c.tomaiuolo@studenti.unipi.it
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

"""Analysis facilities.
"""

import numpy as np

from hexsample.fileio import InputFileBase
from hexsample.hist import Histogram1d
from hexsample.modeling import FitModelBase, DoubleGaussian
from hexsample.plot import plt



# def absz_analysis(input_file : ReconInputFile):
#     """
#     """
#     absz = input_file.mc_column('absz')
#     h = Histogram1d(np.linspace(0., 0.06, 100)).fill(absz)
#     h.plot()
#     setup_gca(logy=True)
#
#
# def cluster_size_analysis(input_file : ReconInputFile):
#     """
#     """
#     clu_size = input_file.column('cluster_size')

def create_histogram(input_file : InputFileBase, column_name : str, mc : bool = False,
    binning: np.ndarray = None, mask : np.ndarray = None) -> Histogram1d:
    """Create a histogram from the values in the given column of the input file.

    This takes either a digi or a recon file as an input and create a one-dimensional
    histogram of the values in a given column.

    Arguments
    ---------
    input_file : DigiInputFile
        The input (digi or recon) file.

    column_name : str
        The name of the column to be histogrammed.

    mc : bool
        If True, histogram a quanity in the MonteCarlo extension of the file.
        Note this must be specified by the user, as the Recon and MonteCarlo tables
        share some of the column names, so that one needs to actively pick one
        or the other.

    binning : array_like or int, optional
        This is following the matplotlib convention, where if ``binning`` is an
        integer, it defines the number of equal-width bins in the range, while if
        it is a sequence, it defines the bin edges, including the left edge of the
        first bin and the right edge of the last bin.

    mask : array_like, optional
        An optional mask on the input values. The length of the mask must match
        that of the values in the input column.
    """
    # pylint: disable=invalid-name
    if mc:
        values = input_file.mc_column(column_name)
    else:
        values = input_file.column(column_name)
    if mask is not None:
        values = values[mask]
    if binning is None:
        binning = 100
    if isinstance(binning, int):
        binning = np.linspace(values.min(), values.max(), binning)
    return Histogram1d(binning, xlabel=column_name).fill(values)

def fit_histogram(hist : Histogram1d, fit_model : FitModelBase = DoubleGaussian,
    p0 : np.array = np.array([1., 8000., 150., 1., 8900., 150.]),
    show_figure : bool = True) -> np.array:
    """Fit an histogram given as argument.

    Arguments
    ---------
    input_file : DigiInputFile
        The input (digi or recon) file.
    hist : Histogram1D
        The histogram to be fitted.
    fitting_model : FitModelBase
        The FitModelBase instance containing the model used for fitting the histogram.
        Default is DoubleGaussian.
    p0 : np.array
        The array containing the initial parameters of the fit. It must have the same
        len() as the number of parameters of the FitModelBase used for the fit.
    plot_figure : bool
        Bool that states if the figure containing the histogram and the best fit has
        to be shown.

    Return
    ------
    model_status : FitModelStatus
        The FitModelStatus instance returned by fit routine.
    """
    # pylint: disable=invalid-name
    model = fit_model()
    model.fit_histogram(hist, p0=p0)
    if show_figure is True:
        hist.plot()
        model.plot()
        model.stat_box()
    return model.status

def double_heatmap(column_vals: np.array, row_vals: np.array , heatmap_values1: np.array,
    heatmap_values2: np.array):
    """Creates a figure containing two different heatmaps (with the same size)
    constructed row by row.

    Even rows (counting from 0) are the rows of the heatmap1 heatmap, odd rows
    are the rows of the heatmap2 heatmap.
    heatmap_values1 and heatmap_values2 must have the same size and share the same axes.
    The resulting figure will have the following dimensions: 2*len(row_vals) x len(column_vals).

    Arguments
    ---------
    column_vals : np.array
        Values relative to the columns of the heatmaps.
    row_vals : np.array
         Values relative to the row of the heatmaps.
    heatmap_values1 : np.array
        A flatten array containing the values of every cell of the first heatmap.
    heatmap_values1 : np.array
        A flatten array containing the values of every cell of the second heatmap.

    Return
    ------
    fig : matplotlib.figure.Figure
        Figure contaning the heatmap.
    ax : matplotlib.axes._axes.Axes
        Axes of fig.
    """
    # pylint: disable=invalid-name, too-many-locals
    #Defining some useful quantities
    column_number = len(column_vals)
    row_number = len(row_vals)
    # Constructing first row of the matrix in order to use np.concatenate() on it.
    heatmap = np.array([heatmap_values1[0:column_number]])
    # Constructing the right np.arange to loop over for constructing the matrix
    # row by row.
    idxes = np.arange(0,row_number)
    # The matrix heatmap contains both heatmaps.
    # Concatenating all rows.
    for i in idxes:
        slice_start = i*column_number
        slice_stop = (i+1)*column_number
        if i == 0:
            tmp = [heatmap_values2[slice_start:slice_stop]]
            heatmap = np.concatenate((heatmap, tmp))
        else:
            heatmap = np.concatenate((heatmap, [heatmap_values1[slice_start:slice_stop]]))
            heatmap = np.concatenate((heatmap, [heatmap_values2[slice_start:slice_stop]]))

    #Creating the custom colormap (it is useful if it is needed to customize it)
    my_cmap = plt.cm.get_cmap('inferno')

    #Plotting the map and make it pretty
    fig = plt.figure()
    ax = plt.gca()
    plt.pcolormesh(heatmap, cmap=my_cmap, edgecolors='k', linewidths=1, shading='flat')
    # Loop over data dimensions and create text annotations.
    fmt = dict(ha='center', va='center', color='b')
    for i in range(row_number * 2):
        for j in range(column_number):
            ax.text(j + 0.5, i + 0.5, f'{heatmap[i, j]:.4f}', **fmt)
    # Shifting ticks on center
    ax.xaxis.set(ticks=np.arange(0.5, column_number), ticklabels=column_vals)
    ax.yaxis.set(ticks=np.arange(1, row_number*2, 2), ticklabels=row_vals)
    fig.tight_layout()
    plt.colorbar() #plotting colorbar
    return fig, ax
