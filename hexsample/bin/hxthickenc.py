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
from hexsample.analysis import pha_analysis
from hexsample.analysis import hist_for_parameter
from hexsample.analysis import hist_fit
from hexsample.analysis import overlapped_pcolormeshes


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
#HXTHICKENC_ARGPARSER.add_infile()

def hxthickenc(**kwargs):
    # Defining the interesting thicknesses and encs to look at. 
    #thickness=np.array([200])
    #enc=np.array([0])
    thickness = np.array([50,100,200,300,500])
    enc = np.array([0,10,20,30,40])
    # Defining arrays where results are contained for all the thick-enc combinations,
    # one for K_alpha (Ka), one for K_beta (Kb).
    # Two arrays: one without cuts on px number, one for 1px events.
    params_matrix = np.empty((len(thickness),len(enc)), dtype=object)
    params_matrix_1px = np.empty((len(thickness),len(enc)), dtype=object)
    #Defining matrix for saving sigma of fit
    sigmas_ka = np.empty((len(thickness),len(enc)))
    sigmas_ka_1px = np.empty((len(thickness),len(enc)))
    sigmas_kb = np.empty((len(thickness),len(enc)))
    sigmas_kb_1px = np.empty((len(thickness),len(enc)))
    #Defining matrix for savign means of fit
    mean_energy_ka = np.empty((len(thickness),len(enc)))
    mean_energy_ka_1px = np.empty((len(thickness),len(enc)))
    mean_energy_kb = np.empty((len(thickness),len(enc)))
    mean_energy_kb_1px = np.empty((len(thickness),len(enc)))

    for thick_idx, thick in np.ndenumerate(thickness):
        for e_idx, e in np.ndenumerate(enc):
            thr = 2 * e
            file_path = f'/Users/chiara/hexsampledata/sim_{thick}um_{e}enc_recon_nn2_thr{thr}.h5'
            recon_file = ReconInputFile(file_path)
            energy_hist_all=hist_for_parameter(recon_file, 'energy', number_of_bins = 100)
            energy_hist_1px=hist_for_parameter(recon_file, 'energy', max_number_of_pixels = 1, number_of_bins = 100)
            fitted_model_all = hist_fit(energy_hist_all, DoubleGaussian, plot_figure=False)
            #Check at which fit parameter is associated the mean of Ka and Kb
            params_matrix[thick_idx][e_idx] = fitted_model_all
            if fitted_model_all.parameter_value('mean0') < fitted_model_all.parameter_value('mean1'):
                # Ka associated to index 0, Kb to index 1
                # filling the matrix with the energy resolution
                sigmas_ka[thick_idx][e_idx] = fitted_model_all.parameter_value('sigma0')
                sigmas_kb[thick_idx][e_idx] = fitted_model_all.parameter_value('sigma1')
                # filling the matrix with the means
                mean_energy_ka[thick_idx][e_idx] = fitted_model_all.parameter_value('mean0')
                mean_energy_kb[thick_idx][e_idx] = fitted_model_all.parameter_value('mean1')
            else:
                # Kb associated to index 0, Ka to index 1
                # filling the matrix with the energy resolution
                sigmas_ka[thick_idx][e_idx] = fitted_model_all.parameter_value('sigma0')
                sigmas_kb[thick_idx][e_idx] = fitted_model_all.parameter_value('sigma1')
                # filling the matrix with the means
                mean_energy_ka[thick_idx][e_idx] = fitted_model_all.parameter_value('mean1')
                mean_energy_kb[thick_idx][e_idx] = fitted_model_all.parameter_value('mean0')

            # Redoing everything for the events with 1px

            fitted_model_1px = hist_fit(energy_hist_1px, DoubleGaussian, plot_figure=False)
            params_matrix_1px[thick_idx][e_idx] = fitted_model_1px
            if fitted_model_1px.parameter_value('mean0') < fitted_model_1px.parameter_value('mean1'):
                # Ka associated to index 0, Kb to index 1
                # filling the matrix with the energy resolution
                sigmas_ka_1px[thick_idx][e_idx] = fitted_model_1px.parameter_value('sigma0')
                sigmas_kb_1px[thick_idx][e_idx] = fitted_model_1px.parameter_value('sigma1')
                # filling the matrix with the means
                mean_energy_ka_1px[thick_idx][e_idx] = fitted_model_all.parameter_value('mean0')
                mean_energy_kb_1px[thick_idx][e_idx] = fitted_model_all.parameter_value('mean1')
            else:
                # Kb associated to index 0, Ka to index 1
                # filling the matrix with the energy resolution
                sigmas_ka_1px[thick_idx][e_idx] = fitted_model_1px.parameter_value('sigma0')
                sigmas_kb_1px[thick_idx][e_idx] = fitted_model_1px.parameter_value('sigma1')
                # filling the matrix with the means
                mean_energy_ka_1px[thick_idx][e_idx] = fitted_model_all.parameter_value('mean1')
                mean_energy_kb_1px[thick_idx][e_idx] = fitted_model_all.parameter_value('mean0')

            
            recon_file.close()
    # After having saved the interesting quantities in arrays, analysis is performed.
    # Saving true energy values (coming from MC).
    mu_true_alpha = 8039.68
    mu_true_beta = 8903.57
    #constructing the metric for the shift of the mean
    z_alpha = 1 - abs(mean_energy_ka-mu_true_alpha)/mu_true_alpha
    z_beta = 1 - abs(mean_energy_kb-mu_true_beta)/mu_true_beta
    #constructing the energy resolution
    res_en_ka = sigmas_ka/mean_energy_ka
    res_en_kb = sigmas_kb/mean_energy_kb

    # Plotting the overlapped heatmaps and customizing them.
    fig,ax = overlapped_pcolormeshes(enc, thickness, z_alpha.flatten(), z_beta.flatten())
    plt.title(r'$\Delta = 1-\frac{|\mu_{E}-E_{K}|}{E_{K}}$, as a function of detector thickness and readout noise')
    plt.ylabel(r'Thickness $\mu$m')
    plt.xlabel('Noise [ENC]')
    # custom yticks. Setting a right yaxis 
    twin1 = ax.twinx()
    twin1.set(ylim=(0, len(enc)*2))
    ticks = []
    for i in range (len(enc)):
        ticks = np.append(ticks, [r'$\alpha$',r'$\beta$'], axis=0)
    twin1.yaxis.set(ticks=np.arange(0.5, len(thickness)*2), ticklabels=ticks)

    fig2,ax2 = overlapped_pcolormeshes(enc, thickness, res_en_ka.flatten(), res_en_kb.flatten())
    plt.title(r'Energy resolution $\frac{\sigma_{E}}{E}$, as a function of detector thickness and readout noise')
    plt.ylabel(r'Thickness $\mu$m')
    plt.xlabel('Noise [ENC]')
    # custom yticks. Setting a right yaxis 
    twin1 = ax2.twinx()
    twin1.set(ylim=(0, len(enc)*2))
    twin1.yaxis.set(ticks=np.arange(0.5, len(thickness)*2), ticklabels=ticks)








    plt.show()

if __name__ == '__main__':
    hxthickenc(**vars(HXTHICKENC_ARGPARSER.parse_args()))