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
from ast import literal_eval

import numpy as np

from hexsample.app import ArgumentParser
from hexsample.fileio import ReconInputFile
from hexsample.modeling import DoubleGaussian
from hexsample.plot import plt
from hexsample.analysis import create_histogram, fit_histogram, double_heatmap, heatmap_with_labels



__description__ = \
""" 
    Preliminary analysis of a grid scan of simulations as a function of:
    - thickness of the silicon detector [mu m];
    - noise of the detector readout [ENC].

    The functions performs the fit with a DoubleGaussian model for all combinations 
    of thickness-noise and plots some interesting quantities for the analysis. 
    
"""

# Parser object.
ANALYZE_GRID_ARGPARSER = ArgumentParser(description=__description__)
ANALYZE_GRID_ARGPARSER.add_argument('1px_ratio_correction', type=str, help='Tells if correcting the\
                                    1px evts with the ratio 1px_evts/tot_evts.\
                                    Accepts True or False')
ANALYZE_GRID_ARGPARSER.add_argument('--savefigpath', type=str, help='Tells whether saving figures\
                                    and where. Accepts an absolute path. Saves figures in pdf.')

def analyze_grid_thickenc(thickness : np.array, enc : np.array, pitch : float, **kwargs) -> None:
    """Opens files, creates histograms, fits them and create figures of some relevant 
        quantities. 

        Arguments
        ---------
        thickness : np.array
            An array containing the thickness values to span
        enc : np.array
            An array containing the enc values to span
        pitch : float
            Pitch value of the simulations
    """
    #Defining the grid of correction factors for 1px events.
    correct_1px_ratio = literal_eval(kwargs['1px_ratio_correction'])
    save_path = kwargs['savefigpath']
    # Defining arrays where results are contained for all thick-enc-pitch combinations,
    # one for K_alpha (Ka), one for K_beta (Kb).
    evt_matrix = np.empty((len(thickness),len(enc)), dtype=object)
    evt_matrix_1px = np.empty((len(thickness),len(enc)), dtype=object)
    #Saving the relative ratio of 1px evts wrt tot evts
    onepx_evts_ratio = np.ones((len(thickness),len(enc)))
    #Defining matrix for saving sigma of fit. 3-dimensional bc every element is a matrix.
    #One for alpha peak fit params, one for beta peaks fit params.
    sigmas = np.empty((2,len(thickness),len(enc)))
    sigmas_1px = np.empty((2,len(thickness),len(enc)))
    #Defining matrix for saving means of fit
    mean_energy = np.empty((2,len(thickness),len(enc)))
    mean_energy_1px = np.empty((2,len(thickness),len(enc)))

    # Saving true energy values (coming from MC).
    mu_true_alpha = 8039.68
    mu_true_beta = 8903.57

    #Opening file, fitting and filling matrices with fitted values
    for thick_idx, thick in np.ndenumerate(thickness):
        for e_idx, e in np.ndenumerate(enc):
            thr = 2 * e
            file_path = f'/Users/chiara/hexsampledata/sim_{thick}um_{e}enc_{pitch}pitch_recon_nn2_thr{thr}.h5'
            recon_file = ReconInputFile(file_path)
            #Constructing the 1px mask
            cluster_size = recon_file.column('cluster_size')
            mask = cluster_size < 2
            #Creating histogram for all events
            energy_hist = create_histogram(recon_file, 'energy', binning = 100)
            fitted_model = fit_histogram(energy_hist, DoubleGaussian, show_figure = False)
            #Creating histogram for events with 1px on readout
            energy_hist_1px = create_histogram(recon_file, 'energy', mask = mask, binning = 100)
            fitted_model_1px = fit_histogram(energy_hist_1px, DoubleGaussian, show_figure = False)
            #Saving the matrix containing the whole FitStatus for further (optional) use
            evt_matrix[thick_idx][e_idx] = fitted_model
            evt_matrix_1px[thick_idx][e_idx] = fitted_model_1px

            #Filling the matrix of sigmas and means
            # filling the matrix with the sigma
            sigmas[0][thick_idx][e_idx] = fitted_model.parameter_value('sigma0') #alpha peak
            sigmas[1][thick_idx][e_idx] = fitted_model.parameter_value('sigma1') #beta peak
            # filling the matrix with the means
            mean_energy[0][thick_idx][e_idx] = fitted_model.parameter_value('mean0')
            mean_energy[1][thick_idx][e_idx] = fitted_model.parameter_value('mean1')

            # Redoing everything for the events with 1px
            #filling the matrix with the means
            mean_energy_1px[0][thick_idx][e_idx] = fitted_model_1px.parameter_value('mean0')
            mean_energy_1px[1][thick_idx][e_idx] = fitted_model_1px.parameter_value('mean1')
            #filling the matrix with the sigma
            sigmas_1px[0][thick_idx][e_idx] = fitted_model_1px.parameter_value('sigma0') #alpha peak
            sigmas_1px[1][thick_idx][e_idx] = fitted_model_1px.parameter_value('sigma1') #beta peak

            #Saving the ratio of 1px wrt all evts if correction is required
            if correct_1px_ratio is True:
                onepx_evts_ratio[thick_idx][e_idx] = energy_hist_1px.content.sum()/energy_hist.content.sum()
            recon_file.close()
    # After having saved the interesting quantities in arrays, analysis is performed.
    #constructing the metric for the shift of the mean
    mean_shift_ka = (mean_energy[0]-mu_true_alpha)/mu_true_alpha
    mean_shift_kb = (mean_energy[1]-mu_true_beta)/mu_true_beta
    #constructing the energy resolution
    energy_res_ka = sigmas[0]/mean_energy[0]
    energy_res_kb = sigmas[1]/mean_energy[1]

    #Repeating for the 1px quantities
    mean_shift_ka_1px = ((mean_energy_1px[0]-mu_true_alpha)/mu_true_alpha)*onepx_evts_ratio
    mean_shift_kb_1px = ((mean_energy_1px[1]-mu_true_beta)/mu_true_beta)*onepx_evts_ratio
    #print(onepx_evts_ratio)
    #constructing the energy resolution
    energy_res_ka_1px = (sigmas_1px[0]/mean_energy_1px[0])*onepx_evts_ratio
    energy_res_kb_1px = (sigmas_1px[1]/mean_energy_1px[1])*onepx_evts_ratio

    # Plotting the overlapped heatmaps and customizing them.
    fig,ax = double_heatmap(enc, thickness, mean_shift_ka.flatten(), mean_shift_kb.flatten())
    plt.figure('mean_shift_all_evts')
    plt.title(r'$\Delta = \frac{\mu_{E}-E_{K}}{E_{K}}$, as a function of\
                detector thickness and readout noise')
    plt.ylabel(r'Thickness $\mu$m')
    plt.xlabel('Noise [ENC]')
    # custom yticks. Setting a right yaxis
    len_yaxis = len(thickness)
    len_yticks = len(thickness)*2
    len_xticks = len(enc)
    twin1 = ax.twinx()
    twin1.set(ylim=(0, len_yaxis))
    ticks = []
    for i in range(len_yaxis):
        ticks = np.append(ticks, [r'$\alpha$',r'$\beta$'], axis=0)
    twin1.yaxis.set(ticks=np.arange(0.25, len_yticks/2, 0.5), ticklabels=ticks)
    if save_path is not None:
        plt.savefig(f'{save_path}/mean_shift_all_evts.pdf')

    fig2,ax2 = double_heatmap(enc, thickness, energy_res_ka.flatten(), energy_res_kb.flatten())
    plt.title(r'Energy resolution $\frac{\sigma_{E}}{E}$, as a function of\
               detector thickness and readout noise')
    plt.ylabel(r'Thickness $\mu$m')
    plt.xlabel('Noise [ENC]')
    # custom yticks. Setting a right yaxis
    twin1 = ax2.twinx()
    twin1.set(ylim=(0, len_yaxis))
    twin1.yaxis.set(ticks=np.arange(0.25, len_yticks/2, 0.5), ticklabels=ticks)
    if save_path is not None:
        plt.savefig(f'{save_path}/energy_res_all_evts.pdf')

    #Repeating everything for 1px
    # Plotting the overlapped heatmaps and customizing them.
    fig,ax = double_heatmap(enc, thickness, mean_shift_ka_1px.flatten(),\
                            mean_shift_kb_1px.flatten())
    plt.title(fr'$\Delta = \frac{{\mu_{{E}}-E_{{K}}}}{{E_{{K}}}}$,\
                for 1px tracks, correction = {correct_1px_ratio}')
    plt.ylabel(r'Thickness $\mu$m')
    plt.xlabel('Noise [ENC]')
    # custom yticks. Setting a right yaxis
    twin1 = ax.twinx()
    twin1.set(ylim=(0, len_yaxis))
    ticks = []
    for i in range (len_yaxis):
        ticks = np.append(ticks, [r'$\alpha$',r'$\beta$'], axis=0)
    twin1.yaxis.set(ticks=np.arange(0.25, len_yticks/2, 0.5), ticklabels=ticks)
    if save_path is not None:
        plt.savefig(f'{save_path}/mean_shift_1px_evts.pdf')
    

    fig2,ax2 = double_heatmap(enc, thickness, energy_res_ka_1px.flatten(), energy_res_kb_1px.flatten())
    plt.title(fr'Energy resolution $\frac{{\sigma_{{E}}}}{{E}}$,\
                for 1px tracks, correction = {correct_1px_ratio}')
    plt.ylabel(r'Thickness $\mu$m')
    plt.xlabel('Noise [ENC]')
    # custom yticks. Setting a right yaxis
    twin1 = ax2.twinx()
    twin1.set(ylim=(0, len_yaxis))
    twin1.yaxis.set(ticks=np.arange(0.25, len_yticks/2, 0.5), ticklabels=ticks)
    if save_path is not None:
        plt.savefig(f'{save_path}/energy_res_1px_evts.pdf')
    

    if correct_1px_ratio is True:
        heatmap_with_labels(enc, thickness, onepx_evts_ratio)
        plt.title(r'Fraction of events with 1 px on readout $f = \frac{n_{evts1px}}{n_{evts}}$')
        plt.xlabel('Noise [ENC]')
        plt.ylabel(r'thickness $\mu$m')
        if save_path is not None:
            plt.savefig(f'{save_path}/fraction_1px_evts.pdf')

    plt.show()


def analyze_grid_thickpitch(thickness : np.array, pitch : np.array, enc : float, n_events : float, **kwargs) -> None:
    """Opens files, creates histograms, fits them and create figures of some relevant 
        quantities. 

        Arguments
        ---------
        thickness : np.array
            An array containing the thickness values to span
        pitch : np.array
            An array containing the pitch values to span
        enc : float
            Enc value in simulations
        n_events : float
            Number of simulated events, it is necessary in order
            to compute the efficiency.
    """
    #Defining the grid of correction factors for 1px events.
    correct_1px_ratio = literal_eval(kwargs['1px_ratio_correction'])
    save_path = kwargs['savefigpath']
    # Defining arrays where results are contained for all thick-enc-pitch combinations,
    # one for K_alpha (Ka), one for K_beta (Kb).
    evt_matrix = np.empty((len(thickness),len(pitch)), dtype=object)
    evt_matrix_1px = np.empty((len(thickness),len(pitch)), dtype=object)
    #Saving the relative ratio of 1px evts wrt tot evts
    onepx_evts_ratio = np.ones((len(thickness),len(pitch)))
    #Defining matrix for saving sigma of fit. 3-dimensional bc every element is a matrix.
    #One for alpha peak fit params, one for beta peaks fit params.
    sigmas = np.empty((2,len(thickness),len(pitch)))
    sigmas_1px = np.empty((2,len(thickness),len(pitch)))
    #Defining matrix for saving means of fit. Same dimensions as sigma above. 
    mean_energy = np.empty((2,len(thickness),len(pitch)))
    mean_energy_1px = np.empty((2,len(thickness),len(pitch)))
    #Defining matrix for saving the quantum efficiency of the detector
    #This is a single number for both peaks so it is a 2D matrix.
    quantum_efficiency = np.empty((len(thickness), len(pitch)))

    # Saving true energy values (coming from MC).
    mu_true_alpha = 8039.68
    mu_true_beta = 8903.57

    #Opening file, fitting and filling matrices with fitted values
    for thick_idx, thick in np.ndenumerate(thickness):
        for p_idx, p in np.ndenumerate(pitch):
            thr = 2 * enc
            file_path = f'/Users/chiara/hexsampledata/sim_{thick}um_{enc}enc_{p}pitch_recon_nn2_thr{thr}.h5'
            recon_file = ReconInputFile(file_path)
            #Constructing the 1px mask
            cluster_size = recon_file.column('cluster_size')
            mask = cluster_size < 2
            #Constructing the efficiency mask: signal events on the overall simulated
            mask_efficiency = cluster_size < 0
            #Creating histogram for all events
            energy_hist = create_histogram(recon_file, 'energy', binning = 100)
            fitted_model = fit_histogram(energy_hist, DoubleGaussian, show_figure = False)
            #Creating histogram for events with 1px on readout
            energy_hist_1px = create_histogram(recon_file, 'energy', mask = mask, binning = 100)
            fitted_model_1px = fit_histogram(energy_hist_1px, DoubleGaussian, show_figure = False)
            #Saving in the matrix the whole FitStatus for further (optional) use
            evt_matrix[thick_idx][p_idx] = fitted_model
            evt_matrix_1px[thick_idx][p_idx] = fitted_model_1px

            #Filling the matrix of sigmas and means
            # filling the matrix with the sigmas
            sigmas[0][thick_idx][p_idx] = fitted_model.parameter_value('sigma0') #alpha peak
            sigmas[1][thick_idx][p_idx] = fitted_model.parameter_value('sigma1') #beta peak
            # filling the matrix with the means
            mean_energy[0][thick_idx][p_idx] = fitted_model.parameter_value('mean0')
            mean_energy[1][thick_idx][p_idx] = fitted_model.parameter_value('mean1')

            # Redoing everything for the events with 1px
            #filling the matrix with the means
            mean_energy_1px[0][thick_idx][p_idx] = fitted_model_1px.parameter_value('mean0')
            mean_energy_1px[1][thick_idx][p_idx] = fitted_model_1px.parameter_value('mean1')
            #filling the matrix with the sigma
            sigmas_1px[0][thick_idx][p_idx] = fitted_model_1px.parameter_value('sigma0') #alpha peak
            sigmas_1px[1][thick_idx][p_idx] = fitted_model_1px.parameter_value('sigma1') #beta peak

            #Constructing the quantum efficiency
            #that is: number of events with nonzero px/number of events simulated
            quantum_efficiency[thick_idx][p_idx] = (n_events-len(cluster_size))/n_events

            #Saving the ratio of 1px wrt all evts if correction is required
            if correct_1px_ratio is True:
                onepx_evts_ratio[thick_idx][p_idx] = energy_hist_1px.content.sum()/energy_hist.content.sum()
            recon_file.close()
    # After having saved the interesting quantities in arrays, analysis is performed.
    print(quantum_efficiency)
    #constructing the metric for the shift of the mean
    mean_shift_ka = (mean_energy[0]-mu_true_alpha)/mu_true_alpha
    mean_shift_kb = (mean_energy[1]-mu_true_beta)/mu_true_beta
    #constructing the energy resolution
    energy_res_ka = sigmas[0]/mean_energy[0]
    energy_res_kb = sigmas[1]/mean_energy[1]

    #Repeating for the 1px quantities
    mean_shift_ka_1px = ((mean_energy_1px[0]-mu_true_alpha)/mu_true_alpha)*onepx_evts_ratio
    mean_shift_kb_1px = ((mean_energy_1px[1]-mu_true_beta)/mu_true_beta)*onepx_evts_ratio
    #print(onepx_evts_ratio)
    #constructing the energy resolution
    energy_res_ka_1px = (sigmas_1px[0]/mean_energy_1px[0])*onepx_evts_ratio
    energy_res_kb_1px = (sigmas_1px[1]/mean_energy_1px[1])*onepx_evts_ratio

    # Plotting the overlapped heatmaps and customizing them.
    fig,ax = double_heatmap(pitch, thickness, mean_shift_ka.flatten(), mean_shift_kb.flatten())
    plt.title(rf'$\Delta = \frac{{\mu_{{E}}-E_{{K}}}}{{E_{{K}}}}$, as a function of detector thickness and readout pitch for enc = {enc}')
    plt.ylabel(r'Thickness [$\mu$m]')
    plt.xlabel(r'Pitch [$\mu$m]')
    # custom yticks. Setting a right yaxis
    len_yaxis = len(thickness)
    len_yticks = len(thickness)*2
    len_xticks = len(pitch)
    twin1 = ax.twinx()
    twin1.set(ylim=(0, len_yaxis))
    ticks = []
    for i in range(len_yaxis):
        ticks = np.append(ticks, [r'$\alpha$',r'$\beta$'], axis=0)
    twin1.yaxis.set(ticks=np.arange(0.25, len_yticks/2, 0.5), ticklabels=ticks)
    if save_path is not None:
            plt.savefig(f'{save_path}/mean_shift_all_evts.pdf')

    fig2,ax2 = double_heatmap(pitch, thickness, energy_res_ka.flatten(), energy_res_kb.flatten())
    plt.title(rf'Energy resolution $\frac{{\sigma_{{E}}}}{{E}}$, as a function of detector thickness and readout pitch for enc = {enc}')
    plt.ylabel(r'Thickness $\mu$m')
    plt.xlabel(r'Pitch [$\mu$m]')
    # custom yticks. Setting a right yaxis
    twin1 = ax2.twinx()
    twin1.set(ylim=(0, len_yaxis))
    twin1.yaxis.set(ticks=np.arange(0.25, len_yticks/2, 0.5), ticklabels=ticks)
    if save_path is not None:
            plt.savefig(f'{save_path}/energy_res_all_evts.pdf')

    #Repeating everything for 1px
    # Plotting the overlapped heatmaps and customizing them.
    fig,ax = double_heatmap(pitch, thickness, mean_shift_ka_1px.flatten(),\
                            mean_shift_kb_1px.flatten())
    plt.title(fr'$\Delta = \frac{{\mu_{{E}}-E_{{K}}}}{{E_{{K}}}}$, for 1px tracks, correction = {correct_1px_ratio} for enc = {enc}')
    plt.ylabel(r'Thickness $\mu$m')
    plt.xlabel(r'Pitch [$\mu$m]')
    # custom yticks. Setting a right yaxis
    twin1 = ax.twinx()
    twin1.set(ylim=(0, len_yaxis))
    ticks = []
    for i in range (len_yaxis):
        ticks = np.append(ticks, [r'$\alpha$',r'$\beta$'], axis=0)
    twin1.yaxis.set(ticks=np.arange(0.25, len_yticks/2, 0.5), ticklabels=ticks)
    if save_path is not None:
            plt.savefig(f'{save_path}/mean_shift_1px_evts.pdf')

    fig2,ax2 = double_heatmap(pitch, thickness, energy_res_ka_1px.flatten(), energy_res_kb_1px.flatten())
    plt.title(fr'Energy resolution $\frac{{\sigma_{{E}}}}{{E}}$, for 1px tracks, correction = {correct_1px_ratio} for enc = {enc}')
    plt.ylabel(r'Thickness $\mu$m')
    plt.xlabel(r'Pitch [$\mu$m]')
    # custom yticks. Setting a right yaxis
    twin1 = ax2.twinx()
    twin1.set(ylim=(0, len_yaxis))
    twin1.yaxis.set(ticks=np.arange(0.25, len_yticks/2, 0.5), ticklabels=ticks)
    if save_path is not None:
            plt.savefig(f'{save_path}/energy_res_1px_evts.pdf')
    

    if correct_1px_ratio is True:
        heatmap_with_labels(pitch, thickness, onepx_evts_ratio)
        plt.title(rf'Fraction of events with 1 px on readout $f = \frac{{n_{{evts1px}}}}{{n_{{evts}}}}$ for enc = {enc}')
        plt.xlabel(r'Pitch [$\mu$m]')
        plt.ylabel(r'thickness $\mu$m')
        if save_path is not None:
            plt.savefig(f'{save_path}/fraction_1px_evts.pdf')

    plt.show()

if __name__ == '__main__':
    #Choosing values of enc and thickness from simulated ones.
    n_events = 100000
    enc_ = np.array([0, 10, 20, 25, 30, 35, 40])
    thickness_ = np.array([0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05])*(1e4)
    pitch = 0.050*1e3
    pitch_ = np.array([0.050, 0.055, 0.060, 0.080, 0.1])*(1e3)
    enc = 0
    #Turning arrays into ints for reading filename correctly
    thickness_ = thickness_.astype(int)
    pitch_ = pitch_.astype(int)
    #analyze_grid_thickenc(thickness_, enc_, pitch, **vars(ANALYZE_GRID_ARGPARSER.parse_args()))
    analyze_grid_thickpitch(thickness_, pitch_, enc, n_events, **vars(ANALYZE_GRID_ARGPARSER.parse_args()))
