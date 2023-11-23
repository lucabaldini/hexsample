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

from typing import Optional, Tuple

from matplotlib.colors import ListedColormap
import numpy as np

from hexsample.fileio import InputFileBase, DigiInputFile, ReconInputFile, FileType
from hexsample.hist import Histogram1d
from hexsample.modeling import FitStatus, FitModelBase, Gaussian, DoubleGaussian
from hexsample.plot import plt, setup_gca



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





def hist_for_parameter(input_file : InputFileBase, parameter_name : str, min_number_of_pixels : int = 0,
    max_number_of_pixels : int = None, number_of_bins : int = 100) -> Histogram1d:
    """This function returns the Histogram1d of the quantity parameter_name.
    It makes distinction between a digi and a recon file.
    For a digi file, the only two interesting quantities to be plotted and analyzed are:
        - roi_size
        - pha
    For a recon file, it is possible to plot every attribute of the recon_table.

    Arguments
    ---------
    input_file : DigiInputFile
        The input (digi or recon) file.

    parameter_name : list[str]
        str containing parameter name of the quantity for which the hist will be returned.

    min_number_of_pixels : int
        gives the minimum (included) number of pixels of the ROI size, so all events
        will have ROI size >= number_of_pixel_cut.

    max_number_of_pixels : int
        gives the maximum (included) number of pixels of the ROI size, so all
        events will have ROI size <= number_of_pixel_cut.

    number_of_bins : int -> number of bins of the output histogram. this is necessary for two main reasons

    Return:
        - Histogram1D of parameter_name with number_of_bins bins.

    Features:
        - Binning is quantity-dependent in order to plot in the best range.
    """
    if input_file.file_type == FileType.DIGI: #digi file type
        # Being those filled in different ways, for this moment using elif statements.
        if parameter_name == 'roi_size':
            rec_quantity = np.array([event.roi.size for event in input_file])
        elif parameter_name == 'energy':
            rec_quantity = np.array([pha.sum() for pha in input_file.pha_array])
        else:
            print(f'No parameter with this name, returning an empty histogram')
            rec_quantity = np.zeros(input_file.NROWS.value)
        range_of_binning = max(rec_quantity) - min(rec_quantity)
        x_right_lim = max(rec_quantity) + np.floor((range_of_binning*0.5)) #using np.floor for having integer limits for integer quantities
        x_left_lim = min(rec_quantity) - np.floor((range_of_binning*0.5))
        binning = np.linspace(x_left_lim, x_right_lim, int(number_of_bins))
        hist = Histogram1d(binning).fill(rec_quantity)
    else: #recon file type
        rec_quantity = input_file.recon_table.col(parameter_name)
        if (max_number_of_pixels != None) or (min_number_of_pixels > 0):
            # creating the mask
            clu_size = input_file.recon_table.col('cluster_size')
            mask_max = clu_size <= max_number_of_pixels
            mask_min = clu_size >= min_number_of_pixels
            mask = np.logical_and(mask_min, mask_max)
            rec_quantity = rec_quantity[mask]  #masking the quantity
            print(rec_quantity)
        range_of_binning = max(rec_quantity) - min(rec_quantity)
        x_right_lim = max(rec_quantity) + np.floor((range_of_binning*0.5)) #using np.floor for having integer limits for integer quantities
        x_left_lim = min(rec_quantity) - np.floor((range_of_binning*0.5))
        binning = np.linspace(x_left_lim, x_right_lim, int(number_of_bins))
        hist = Histogram1d(binning).fill(rec_quantity)

    return hist

def hist_fit(hist : Histogram1d, fit_model : FitModelBase = DoubleGaussian, p0 : np.array = np.array([1., 8000., 150., 1., 8900., 150.]), plot_figure : bool = True) -> np.array:
    """
        This function does the fit to an histogram
        Input parameters:
            - hist : Histogram1d, an histogram to be fitted;
            - fit_model : FitModelBase = DoubleGaussian, an instance of subclass of FitModelBase;
            - p0 : np.array = np.array([1., 8000., 150., 1., 8900., 150.]), array of initial parameters for fit;
            - plot_figure : bool = True, bool that states if the figure containing the histogram and the best fit has to be shown.
        Return:
            - model_status : FitModelStatus, FitModelStatus instance of hist fit.
    """
    if plot_figure == True:
        hist.plot()
    model = fit_model()
    model.fit_histogram(hist, p0=p0, xmin = hist.binning[0][0], xmax = hist.binning[0][-1])
    model_status=model.status
    if plot_figure == True:
        model.plot()
        model.stat_box()
    return model_status

def pha_analysis(input_file : ReconInputFile, PHA_cut_value : float=0) -> Tuple[Histogram1d]:
    """
        This functions does some analysis on the attribute PHA of a recon file.
        Input parameters:
            - input_file : ReconInputFile, the recon file to analyze;
            - PHA_cut_value : float = 0, a cut value for PHA values (default is 0)
        Return:
            - h_clu_size : Histogram1D, histogram of cluster size;
            - h_energy_tot : Histogram1D, histogram of PHA

        Features:
        - Binning is file-dependent in order to plot in the best range;
        - !To be discussed and possibly implemented! It is possible to set a cut value for PHA values (default is 0);
    """
    rec_energy = input_file.recon_table.col('energy')
    clu_size = input_file.recon_table.col('cluster_size')

    plt.figure('Cluster size')
    plt.xlabel('Cluster size [number of pixels]')
    x_right_lim=max(clu_size)+2
    x_left_lim=max(min(clu_size)-2, 0)
    number_of_bins=(x_right_lim-x_left_lim)+1
    binning_clu_size=np.linspace(x_left_lim, x_right_lim, number_of_bins)
    h_clu_size = Histogram1d(binning_clu_size).fill(clu_size)
    h_clu_size.plot()
    #Creating energy plot for all spectrum
    plt.figure('Energy spectrum')
    binning_energy = np.linspace(rec_energy.min(), rec_energy.max(), 100)
    h_energy_tot = Histogram1d(binning_energy).fill(rec_energy)
    h_energy_tot.plot()
    setup_gca(xlabel='Energy [keV]')
    #Inserting a parameter for choosing the fit model? Passing fit model function?
    #Default is sum of two Gaussian pdfs?
    #gauss_model = DoubleGaussian()
    gauss_model = Gaussian() + Gaussian()
    gauss_model.fit_histogram(h_energy_tot, p0=(1., 8000., 150., 1., 8900., 150.), xmin = rec_energy.min(), xmax = rec_energy.max())
    gauss_model.plot()
    gauss_model.stat_box()
    #Computing the selection efficiency
    mask = clu_size <= 1 #those are the events that did not trigger the chip or that triggered just one pixel
    frac = mask.sum() / len(mask)
    print(f'Selection efficiency = {frac}')
    rec_energy = rec_energy[mask] #Recorded energy for single-pixel events

    return h_clu_size,h_energy_tot


def overlapped_pcolormeshes(x_values : np.array, y_values : np.array , to_map1 : np.array, to_map2 : np.array):
    '''
        This function is used for plotting two plt.pcolormesh with the same dimension
        and axes in the same figure.
        Input parameters:
            - x_values : np.array, column values (x-axis values);
            - y_values : np.array, row values (y-axis values);
            - to_map1 : np.array, the first (flattened matrix) array to map;
            - to_map2 : np.array, the second (flattened matrix) array to map;
        Return:
            - fig : matplotlib.figure.Figure, figure containing heatmap;
            - ax : matplotlib.axes._axes.Axes, axes of fig.
        Method:
            Data are stacked row by row (adjacently, so every adjacent row correspond to the same
            y_value), resulting in an heatmap having dimension 2*len(y_value) x len(x_value).
            Values of every cell are pront on heatmap for clarity.
            The default colormap is 'inferno' and text color is 'b'.
            Figure is returned for further external customization.
    '''
    # Constructing first row of the matrix in order to use np.concatenate() on it.
    to_map_tot=np.array([to_map1[0:len(y_values)]])
    # Constructing the right np.arange to loop over for constructing the matrix
    # row by row.
    # The matrix to_map_tot contains all values to be plotted.
    idxes = np.arange(0,len(y_values))
    # Concatenating all rows.
    for i in idxes:
        if i == 0:
            tmp = [to_map2[i*len(x_values):(i+1)*len(x_values)]]
            to_map_tot = np.concatenate((to_map_tot, tmp),axis=0)
        else:
            row_map1 = [to_map1[i*len(x_values):(i+1)*len(x_values)]]
            row_map2 = [to_map2[i*len(x_values):(i+1)*len(x_values)]]
            to_map_tot = np.concatenate((to_map_tot, row_map1),axis=0)
            to_map_tot = np.concatenate((to_map_tot, row_map2),axis=0)

    #Creating the colormap (it is useful if it is needed to customize it)
    my_cmap = plt.cm.get_cmap('inferno')

    # Everything is now ready for plotting
    fig = plt.figure()
    ax = plt.gca()
    plt.pcolormesh(to_map_tot, cmap=my_cmap, edgecolors='k', linewidths=1,shading='flat')
    # Loop over data dimensions and create text annotations.
    for i in range(len(y_values)*2):
        for j in range(len(x_values)):
            text = ax.text(j+0.5, i+0.5, "{:.4f}".format(to_map_tot[i, j]),
                        ha="center", va="center", color="b")
    # Shifting ticks on center
    ax.xaxis.set(ticks=np.arange(0.5, len(x_values)), ticklabels=x_values)
    ax.yaxis.set(ticks=np.arange(1, len(y_values)*2, 2), ticklabels=y_values)
    fig.tight_layout()
    plt.colorbar() #plotting colorbar
    return fig,ax
