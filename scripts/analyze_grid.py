#!/usr/bin/env python
#
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

"""Event file viewer.
"""

import numpy as np

from hexsample.app import ArgumentParser
from hexsample.modeling import FitModelBase
from hexsample.fitting import fit_gaussian_iterative
from hexsample.hist import Histogram1d
from hexsample.fileio import ReconInputFile
from hexsample.modeling import Gaussian, DoubleGaussian
from hexsample.plot import plt
from hexsample.analysis import create_histogram, fit_histogram, double_heatmap



__description__ = \
""" 
    Preliminary analysis of a grid scan of simulations as a function of:
    - thickness of the silicon detector [\mu m];
    - noise of the detector readout [ENC].

    The functions performs the fit with a DoubleGaussian model for all combinations 
    of thickness-noise and plots some interesting quantities for the analysis. 
    
"""

# Parser object.
HXTHICKENC_ARGPARSER = ArgumentParser(description=__description__)

def hxthickenc(thickness : np.array, enc : np.array, **kwargs) -> None:
    """Opens files, creates histograms, fits them and create figures of some relevant 
        quantities. 

        Arguments
        ---------
        thickness : np.array
            An array containing the thickness values to span
        enc : np.array
            An array containing the enc values to span
    """
    # Defining arrays where results are contained for all the thick-enc combinations,
    # one for K_alpha (Ka), one for K_beta (Kb).
    # Two arrays: one without cuts on px number, one for 1px events.
    params_matrix = np.empty((len(thickness),len(enc)), dtype=object)
    params_matrix_1px = np.empty((len(thickness),len(enc)), dtype=object)
    #Defining matrix for saving sigma of fit. 3-dimensional bc every element is a matrix.
    #One for alpha peak fit params, one for beta peaks fit params. 
    sigmas = np.empty((2,len(thickness),len(enc)))
    sigmas_1px = np.empty((2,len(thickness),len(enc)))
    #Defining matrix for saving means of fit
    mean_energy = np.empty((2,len(thickness),len(enc)))
    mean_energy_1px = np.empty((2,len(thickness),len(enc)))

    #Opening file, fitting and filling matrices with fitted values
    for thick_idx, thick in np.ndenumerate(thickness):
        for e_idx, e in np.ndenumerate(enc):
            thr = 2 * e
            file_path = f'/Users/chiara/hexsampledata/sim_{thick}um_{e}enc_recon_nn2_thr{thr}.h5'
            recon_file = ReconInputFile(file_path)
            #Constructing the 1px mask 
            cluster_size = recon_file.column('cluster_size')
            mask = cluster_size < 2
            energy_hist = create_histogram(recon_file, 'energy', binning = 100)
            energy_hist_1px = create_histogram(recon_file, 'energy', mask = mask, binning = 100)
            fitted_model = fit_histogram(energy_hist, DoubleGaussian, show_figure = False)
            fitted_model_1px = fit_histogram(energy_hist_1px, DoubleGaussian, show_figure = False)
            #Check at which fit parameter is associated the mean of Ka and Kb
            params_matrix[thick_idx][e_idx] = fitted_model
            if fitted_model.parameter_value('mean0') < fitted_model.parameter_value('mean1'):
                # Ka associated to index 0, Kb to index 1
                # filling the matrix with the energy resolution
                sigmas[0][thick_idx][e_idx] = fitted_model.parameter_value('sigma0') #alpha peak
                sigmas[1][thick_idx][e_idx] = fitted_model.parameter_value('sigma1') #beta peak
                # Redoing everything for the events with 1px
                sigmas_1px[0][thick_idx][e_idx] = fitted_model_1px.parameter_value('sigma0') #alpha peak
                sigmas_1px[1][thick_idx][e_idx] = fitted_model_1px.parameter_value('sigma1') #beta peak
                # filling the matrix with the means
                mean_energy[0][thick_idx][e_idx] = fitted_model.parameter_value('mean0')
                mean_energy[1][thick_idx][e_idx] = fitted_model.parameter_value('mean1')
                # Redoing everything for the events with 1px
                mean_energy_1px[0][thick_idx][e_idx] = fitted_model_1px.parameter_value('mean0')
                mean_energy_1px[1][thick_idx][e_idx] = fitted_model_1px.parameter_value('mean1')
            else:
                # Kb associated to index 0, Ka to index 1
                # filling the matrix with the energy resolution
                sigmas[1][thick_idx][e_idx] = fitted_model.parameter_value('sigma0')
                sigmas[0][thick_idx][e_idx] = fitted_model.parameter_value('sigma1')
                # Redoing everything for the events with 1px
                sigmas_1px[1][thick_idx][e_idx] = fitted_model_1px.parameter_value('sigma0')
                sigmas_1px[0][thick_idx][e_idx] = fitted_model_1px.parameter_value('sigma1')
                # filling the matrix with the means
                mean_energy[1][thick_idx][e_idx] = fitted_model.parameter_value('mean0')
                mean_energy[0][thick_idx][e_idx] = fitted_model.parameter_value('mean1')
                # Redoing everything for the events with 1px
                mean_energy_1px[1][thick_idx][e_idx] = fitted_model_1px.parameter_value('mean0')
                mean_energy_1px[0][thick_idx][e_idx] = fitted_model_1px.parameter_value('mean1')
            
            recon_file.close()
    # After having saved the interesting quantities in arrays, analysis is performed.
    # Saving true energy values (coming from MC).
    mu_true_alpha = 8039.68
    mu_true_beta = 8903.57
    #constructing the metric for the shift of the mean
    mean_shift_ka = 1 - abs(mean_energy[0]-mu_true_alpha)/mu_true_alpha
    mean_shift_kb = 1 - abs(mean_energy[0]-mu_true_beta)/mu_true_beta
    #constructing the energy resolution
    energy_res_ka = sigmas[0]/mean_energy[0]
    energy_res_kb = sigmas[1]/mean_energy[1]

    #Repeating for the 1px quantities
    mean_shift_ka_1px = 1 - abs(mean_energy_1px[0]-mu_true_alpha)/mu_true_alpha
    mean_shift_kb_1px = 1 - abs(mean_energy_1px[0]-mu_true_beta)/mu_true_beta
    #constructing the energy resolution
    energy_res_ka_1px = sigmas_1px[0]/mean_energy_1px[0]
    energy_res_kb_1px = sigmas_1px[1]/mean_energy_1px[1]

    # Plotting the overlapped heatmaps and customizing them.
    fig,ax = double_heatmap(enc, thickness, mean_shift_ka.flatten(), mean_shift_kb.flatten())
    plt.title(r'$\Delta = 1-\frac{|\mu_{E}-E_{K}|}{E_{K}}$, as a function of detector thickness and readout noise')
    plt.ylabel(r'Thickness $\mu$m')
    plt.xlabel('Noise [ENC]')
    # custom yticks. Setting a right yaxis 
    len_yaxis = len(thickness)
    len_yticks = len(thickness)*2
    len_xticks = len(enc)
    twin1 = ax.twinx()
    twin1.set(ylim=(0, len_yaxis))
    ticks = []
    for i in range (len_yaxis):
        ticks = np.append(ticks, [r'$\alpha$',r'$\beta$'], axis=0)
    twin1.yaxis.set(ticks=np.arange(0.5, len_yticks), ticklabels=ticks)

    fig2,ax2 = double_heatmap(enc, thickness, energy_res_ka.flatten(), energy_res_kb.flatten())
    plt.title(r'Energy resolution $\frac{\sigma_{E}}{E}$, as a function of detector thickness and readout noise')
    plt.ylabel(r'Thickness $\mu$m')
    plt.xlabel('Noise [ENC]')
    # custom yticks. Setting a right yaxis 
    twin1 = ax2.twinx()
    twin1.set(ylim=(0, len_yaxis))
    twin1.yaxis.set(ticks=np.arange(0.5, len_yticks), ticklabels=ticks)

    #Repeating everything for 1px 
    # Plotting the overlapped heatmaps and customizing them.
    fig,ax = double_heatmap(enc, thickness, mean_shift_ka_1px.flatten(), mean_shift_kb_1px.flatten())
    plt.title(r'$\Delta = 1-\frac{|\mu_{E}-E_{K}|}{E_{K}}$, as a function of detector thickness and readout noise for 1px tracks')
    plt.ylabel(r'Thickness $\mu$m')
    plt.xlabel('Noise [ENC]')
    # custom yticks. Setting a right yaxis 
    twin1 = ax.twinx()
    twin1.set(ylim=(0, len_yaxis))
    ticks = []
    for i in range (len_yaxis):
        ticks = np.append(ticks, [r'$\alpha$',r'$\beta$'], axis=0)
    twin1.yaxis.set(ticks=np.arange(0.5, len_yticks), ticklabels=ticks)

    fig2,ax2 = double_heatmap(enc, thickness, energy_res_ka_1px.flatten(), energy_res_kb_1px.flatten())
    plt.title(r'Energy resolution $\frac{\sigma_{E}}{E}$, as a function of detector thickness and readout noise for 1px tracks')
    plt.ylabel(r'Thickness $\mu$m')
    plt.xlabel('Noise [ENC]')
    # custom yticks. Setting a right yaxis 
    twin1 = ax2.twinx()
    twin1.set(ylim=(0, len_yaxis))
    twin1.yaxis.set(ticks=np.arange(0.5, len_yticks), ticklabels=ticks)



    plt.show()

if __name__ == '__main__':
    #Choosing values of enc and thickness from simulated ones. 
    enc = np.array([20, 25, 30, 35, 40, 45, 50])
    thickness = np.array([0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06])*(1e4)
    #Turning array into ints for reading filename correctly
    thickness = thickness.astype(int)
    #hxthickenc(np.array([50,100,200,300,500]), np.array([0,10,20,30,40]), **vars(HXTHICKENC_ARGPARSER.parse_args()))
    hxthickenc(thickness, enc, **vars(HXTHICKENC_ARGPARSER.parse_args()))