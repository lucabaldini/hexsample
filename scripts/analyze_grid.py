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
from hexsample.analysis import create_histogram, fit_histogram, double_heatmap, heatmap_with_labels, energy_threshold_computation
from hexsample.fileio import ReconInputFile
from hexsample.hist import Histogram1d
from hexsample.modeling import DoubleGaussian
from hexsample.plot import plt
from hexsample.sensor import Silicon, SiliconSensor


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
ANALYZE_GRID_ARGPARSER.add_argument('only_1px_evts', type=str, help='Tells if only 1px.\
                                    evts are taken int consideration. Accepts True or False')
ANALYZE_GRID_ARGPARSER.add_argument('1px_ratio_correction', type=str, help='Tells if correcting the\
                                    1px evts with the ratio 1px_evts/tot_evts.\
                                    Accepts True or False')
ANALYZE_GRID_ARGPARSER.add_argument('--savefigpath', type=str, help='Tells whether saving figures\
                                    and where. Accepts an absolute path. Saves figures in pdf.')


def analyze_grid_zthr(thickness: float, enc: float, pitch: float, zthreshold: np.array, nneighbors: np.array):
    # Defining arrays where results are contained for all zthr-nn combinations,
    # one for K_alpha (Ka), one for K_beta (Kb).
    evt_matrix = np.empty((len(nneighbors),len(zthreshold)), dtype=object)
    #Defining matrix for saving sigma of fit. 3-dimensional bc every element is a matrix.
    #One for alpha peak fit params, one for beta peaks fit params.
    sigmas = np.empty((2,len(nneighbors),len(zthreshold)))
    #Defining matrix for saving means of fit
    mean_energy = np.empty((2,len(nneighbors),len(zthreshold)))
    #Creating a matrix for saving the mean cluster size
    mean_cluster_size = np.empty((len(nneighbors),len(zthreshold)))
    #Creating a matrix for saving the efficiency on alpha signal
    efficiency_on_alpha = np.empty((len(nneighbors),len(zthreshold)))
    #Creating matrices for saving the mean difference x_mc - x_reco and y_mc - y_reco
    dx = np.empty((len(nneighbors),len(zthreshold)))
    dy = np.empty((len(nneighbors),len(zthreshold)))

    # Saving true energy values (coming from MC).
    mu_true_alpha = 8039.68
    mu_true_beta = 8903.57

    #Opening file, fitting and filling matrices with fitted values
    for zthr_idx, zthr in np.ndenumerate(zthreshold):
        for nn_idx, nn in np.ndenumerate(nneighbors):
            file_path = f'/Users/chiara/hexsampledata/sim_{thickness}um_{enc}enc_{pitch}pitch_recon_nn{nn}_thr{zthr:.0f}.h5'
            recon_file = ReconInputFile(file_path)
            energy_hist = create_histogram(recon_file, 'energy', binning = 100)
            fitted_model = fit_histogram(energy_hist, DoubleGaussian, show_figure = False)
            #Saving the matrix containing the whole FitStatus for further (optional) use
            evt_matrix[nn_idx][zthr_idx] = fitted_model
            
            #Filling the matrix of sigmas and means
            # filling the matrix with the sigma
            sigmas[0][nn_idx][zthr_idx] = fitted_model.parameter_value('sigma0') #alpha peak
            sigmas[1][nn_idx][zthr_idx] = fitted_model.parameter_value('sigma1') #beta peak
            # filling the matrix with the means
            mean_energy[0][nn_idx][zthr_idx] = fitted_model.parameter_value('mean0')
            mean_energy[1][nn_idx][zthr_idx] = fitted_model.parameter_value('mean1')
            # filling the matrices containing the true and reconstructed x and y
            dx[nn_idx][zthr_idx] = np.mean(np.abs(recon_file.mc_column('absx') - recon_file.column('posx')))
            dy[nn_idx][zthr_idx] = np.mean(np.abs(recon_file.mc_column('absy') - recon_file.column('posy')))
            recon_file.close()

    # After having saved the interesting quantities in arrays, analysis is performed.
    #constructing the metric for the shift of the mean
    mean_shift_ka = (mean_energy[0]-mu_true_alpha)/mu_true_alpha
    mean_shift_kb = (mean_energy[1]-mu_true_beta)/mu_true_beta
    #constructing the energy resolution
    energy_res_ka = sigmas[0]*2.35 #eV (FWHM)
    energy_res_kb = sigmas[1]*2.35 #eV (FWHM)

    # Plotting the energy resolution vs the z sup threshold
    plt.figure("Energy resolution vs zero suppression threshold")
    plt.xlabel('Zero suppression threshold [enc]')
    plt.ylabel(rf'Energy resolution $\sigma_{{K_{{\alpha, \beta}}}}$ FWHM [eV]')
    for nn_idx, nn in np.ndenumerate(nneighbors):
        plt.errorbar(zthreshold, energy_res_ka[:][nn_idx], label=f'NN = {nn}')
    plt.legend()
    
        

def analyze_grid_thickenc(thickness : np.array, enc : np.array, pitch : float, onepx : bool=False, **kwargs) -> None:
    """Opens files, creates histograms, fits them and create figures of some relevant 
        quantities. 

        Arguments
        ---------
        -thickness : np.array
            An array containing the thickness values to span
        -enc : np.array
            An array containing the enc values to span
        -pitch : float
            Pitch value of the simulations
        -onepx : bool
            States if analyze only events with 1px (True) or all (False).
    """
    #Defining the grid of correction factors for 1px events.
    onepx = literal_eval(kwargs['only_1px_evts'])
    correct_1px_ratio = literal_eval(kwargs['1px_ratio_correction'])
    save_path = kwargs['savefigpath']
    # Defining arrays where results are contained for all thick-enc-pitch combinations,
    # one for K_alpha (Ka), one for K_beta (Kb).
    evt_matrix = np.empty((len(thickness),len(enc)), dtype=object)
    #Saving the relative ratio of 1px evts wrt tot evts
    onepx_evts_ratio = np.ones((len(thickness),len(enc)))
    #Defining matrix for saving sigma of fit. 3-dimensional bc every element is a matrix.
    #One for alpha peak fit params, one for beta peaks fit params.
    sigmas = np.empty((2,len(thickness),len(enc)))
    #Defining matrix for saving means of fit
    mean_energy = np.empty((2,len(thickness),len(enc)))
    #Creating a matrix for saving the mean cluster size
    mean_cluster_size = np.empty((len(thickness), len(enc)))
    #Creating a matrix for saving the efficiency on alpha signal
    efficiency_on_alpha = np.empty((len(thickness), len(enc)))
    #Creating matrices for saving the mean difference x_mc - x_reco and y_mc - y_reco
    dx = np.empty((len(thickness), len(enc)))
    dy = np.empty((len(thickness), len(enc)))

    # Saving true energy values (coming from MC).
    mu_true_alpha = 8039.68
    mu_true_beta = 8903.57

    #Opening file, fitting and filling matrices with fitted values
    for thick_idx, thick in np.ndenumerate(thickness):
        sensor = SiliconSensor(thick*1e-4)
        QE_alpha = sensor.photabsorption_efficiency(np.mean(8039.68))
        QE_beta = sensor.photabsorption_efficiency(np.mean(8903.57))
        for e_idx, e in np.ndenumerate(enc):
            thr = 2 * e
            file_path = f'/Users/chiara/hexsampledata/sim_{thick}um_{e}enc_{pitch}pitch_recon_nn2_thr{thr}.h5'
            recon_file = ReconInputFile(file_path)
            #Constructing the 1px mask
            cluster_size = recon_file.column('cluster_size')
            onepxmask = cluster_size < 2
            #Creating histogram
            if onepx is False:
                mask = cluster_size > 0
            else:
                mask = onepxmask
            energy_hist = create_histogram(recon_file, 'energy', mask = mask, binning = 100)
            fitted_model = fit_histogram(energy_hist, DoubleGaussian, show_figure = False)
            #Saving the matrix containing the whole FitStatus for further (optional) use
            evt_matrix[thick_idx][e_idx] = fitted_model
            
            #Filling the matrix of sigmas and means
            # filling the matrix with the sigma
            sigmas[0][thick_idx][e_idx] = fitted_model.parameter_value('sigma0') #alpha peak
            sigmas[1][thick_idx][e_idx] = fitted_model.parameter_value('sigma1') #beta peak
            # filling the matrix with the means
            mean_energy[0][thick_idx][e_idx] = fitted_model.parameter_value('mean0')
            mean_energy[1][thick_idx][e_idx] = fitted_model.parameter_value('mean1')
            # filling the matrices containing the true and reconstructed x and y
            dx[thick_idx][e_idx] = np.mean(np.abs(recon_file.mc_column('absx') - recon_file.column('posx')))
            dy[thick_idx][e_idx] = np.mean(np.abs(recon_file.mc_column('absy') - recon_file.column('posy')))
            #Computing energy threshold for classification and relative efficiency
            energy_thr, efficiency = energy_threshold_computation(fitted_model) #contamination set to 2%
            efficiency_on_alpha[thick_idx][e_idx] = efficiency*QE_alpha
            #Constructing the mean cluster size
            mean_cluster_size[thick_idx][e_idx] = np.mean(cluster_size[mask])

            #Saving the ratio of 1px wrt all evts if correction is required
            if correct_1px_ratio is True:
                onepx_evts_ratio[thick_idx][e_idx] = len(cluster_size[onepxmask])/len(cluster_size)
            recon_file.close()
    # After having saved the interesting quantities in arrays, analysis is performed.
    #constructing the metric for the shift of the mean
    mean_shift_ka = (mean_energy[0]-mu_true_alpha)/mu_true_alpha
    mean_shift_kb = (mean_energy[1]-mu_true_beta)/mu_true_beta
    #constructing the energy resolution
    energy_res_ka = sigmas[0]*2.35 #eV (FWHM)
    energy_res_kb = sigmas[1]*2.35 #eV (FWHM)

    #print(mean_cluster_size)
    #print(efficiency_on_alpha)

    # Plotting the overlapped heatmaps and customizing them.
    fig,ax = double_heatmap(enc, thickness, mean_shift_ka.flatten(), mean_shift_kb.flatten())
    plt.title(fr'$\Delta = \frac{{\mu_{{E}}-E_{{K}}}}{{E_{{K}}}}$, p = {pitch} $\mu$m')
    plt.ylabel(r'Thickness t $\mu$m')
    plt.xlabel(r'Electronic noise per pixel $n_{electronics}$ [ENC]')
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

    fig2,ax2 = double_heatmap(enc, thickness, energy_res_ka.flatten(), energy_res_kb.flatten(), 0)
    plt.title(rf'Energy resolution $\sigma_{{K_{{\alpha, \beta}}}}$ FWHM [eV], p = {pitch} $\mu$m')
    plt.ylabel(r'Thickness t $\mu$m')
    plt.xlabel(r'Electronic noise per pixel $n_{electronics}$ [ENC]')
    # custom yticks. Setting a right yaxis
    twin1 = ax2.twinx()
    twin1.set(ylim=(0, len_yaxis))
    twin1.yaxis.set(ticks=np.arange(0.25, len_yticks/2, 0.5), ticklabels=ticks)
    if save_path is not None:
        plt.savefig(f'{save_path}/energy_res_all_evts.pdf')

    heatmap_with_labels(enc, thickness, mean_cluster_size, 2)
    plt.title(rf'Average cluster size,  p = {pitch} $\mu$m')
    plt.xlabel(r'Electronic noise per pixel $n_{electronics}$ [ENC]')
    plt.ylabel(r'Thickness t $\mu$m')
    if save_path is not None:
        plt.savefig(f'{save_path}/mean_cluster_size.pdf')

    heatmap_with_labels(enc, thickness, efficiency_on_alpha)
    plt.title(rf'Efficiency on $K_{{\alpha}}$ signal,  p = {pitch} $\mu$m')
    plt.xlabel(r'Electronic noise per pixel $n_{electronics}$ [ENC]')
    plt.ylabel(r'Thickness t $\mu$m')
    if save_path is not None:
        plt.savefig(f'{save_path}/efficiency.pdf')

    heatmap_with_labels(enc, thickness, dx, n_digits=7)
    plt.title(rf'Reconstruction shift on x: $\mu(|x_{{MC}} - x_{{recon}}|)$ [cm],  p = {pitch} $\mu$m')
    plt.xlabel(r'Electronic noise per pixel $n_{electronics}$ [ENC]')
    plt.ylabel(r'Thickness t $\mu$m')
    if save_path is not None:
        plt.savefig(f'{save_path}/x_shift.pdf')

    heatmap_with_labels(enc, thickness, dy, n_digits=7)
    plt.title(rf'Reconstruction shift on y: $\mu(|y_{{MC}} - y_{{recon}}|)$ [cm],  p = {pitch} $\mu$m')
    plt.xlabel(r'Electronic noise per pixel $n_{electronics}$ [ENC]')
    plt.ylabel(r'Thickness t $\mu$m')
    if save_path is not None:
        plt.savefig(f'{save_path}/y_shift.pdf')

    if correct_1px_ratio is True:
        heatmap_with_labels(enc, thickness, onepx_evts_ratio)
        plt.title(fr'Fraction of events with 1 px on readout $f = \frac{{n_{{evts1px}}}}{{n_{{evts}}}}$,  p = {pitch} $\mu$m')
        plt.xlabel(r'Electronic noise per pixel $n_{electronics}$ [ENC]')
        plt.ylabel(r'Thickness t $\mu$m')
        if save_path is not None:
            plt.savefig(f'{save_path}/fraction_1px_evts.pdf')

    plt.show()


def analyze_grid_thickpitch(thickness : np.array, pitch : np.array, enc : float, onepx : bool=False, **kwargs) -> None:
    """Opens files, creates histograms, fits them and create figures of some relevant 
        quantities. 

        Arguments
        ---------
        -thickness : np.array
            An array containing the thickness values to span
        -pitch : np.array
            An array containing the pitch values to span
        -enc : float
            Enc value in simulations
        -onepx : bool
            States if analyze only events with 1px (True) or all (False).
    """
    #Defining the grid of correction factors for 1px events.
    onepx = literal_eval(kwargs['only_1px_evts'])
    correct_1px_ratio = literal_eval(kwargs['1px_ratio_correction'])
    save_path = kwargs['savefigpath']
    # Defining arrays where results are contained for all thick-enc-pitch combinations,
    # one for K_alpha (Ka), one for K_beta (Kb).
    evt_matrix = np.empty((len(thickness),len(pitch)), dtype=object)
    #Saving the relative ratio of 1px evts wrt tot evts
    onepx_evts_ratio = np.ones((len(thickness),len(pitch)))
    #Defining matrix for saving sigma of fit. 3-dimensional bc every element is a matrix.
    #One for alpha peak fit params, one for beta peaks fit params.
    sigmas = np.empty((2,len(thickness),len(pitch)))
    #Defining matrix for saving means of fit. Same dimensions as sigma above. 
    mean_energy = np.empty((2,len(thickness),len(pitch)))
    #Defining matrix for saving the quantum efficiency of the detector
    #This is a single number for both peaks so it is a 2D matrix.
    mean_cluster_size = np.empty((len(thickness), len(pitch)))
    efficiency_on_alpha = np.empty((len(thickness), len(pitch)))
    #Creating matrices for saving the mean difference x_mc - x_reco and y_mc - y_reco
    dx = np.empty((len(thickness), len(pitch)))
    dy = np.empty((len(thickness), len(pitch)))

    # Saving true energy values (coming from MC).
    mu_true_alpha = 8039.68
    mu_true_beta = 8903.57

    #Opening file, fitting and filling matrices with fitted values
    for thick_idx, thick in np.ndenumerate(thickness):
        sensor = SiliconSensor(thick*1e-4)
        QE_alpha = sensor.photabsorption_efficiency(8039.68)
        print(f'Thickness:{thick}, QE={QE_alpha}')
        QE_beta = sensor.photabsorption_efficiency(8903.57)
        for p_idx, p in np.ndenumerate(pitch):
            thr = 2 * enc
            file_path = f'/Users/chiara/hexsampledata/sim_{thick}um_{enc}enc_{p}pitch_recon_nn2_thr{thr}.h5'
            recon_file = ReconInputFile(file_path)
            #Constructing the 1px mask
            cluster_size = recon_file.column('cluster_size')
            onepxmask = cluster_size < 2
            #Creating histogram
            if onepx is False:
                mask = cluster_size > 0
            else:
                mask = onepxmask
            energy_hist = create_histogram(recon_file, 'energy', mask = mask, binning = 100)
            fitted_model = fit_histogram(energy_hist, DoubleGaussian, show_figure = False)
            #Saving in the matrix the whole FitStatus for further (optional) use
            evt_matrix[thick_idx][p_idx] = fitted_model

            #Filling the matrix of sigmas and means
            # filling the matrix with the sigmas
            sigmas[0][thick_idx][p_idx] = fitted_model.parameter_value('sigma0') #alpha peak
            sigmas[1][thick_idx][p_idx] = fitted_model.parameter_value('sigma1') #beta peak
            # filling the matrix with the means
            mean_energy[0][thick_idx][p_idx] = fitted_model.parameter_value('mean0')
            mean_energy[1][thick_idx][p_idx] = fitted_model.parameter_value('mean1')
            # filling the matrices containing the true and reconstructed x and y
            dx[thick_idx][p_idx] = np.mean(np.abs(recon_file.mc_column('absx') - recon_file.column('posx')))
            dy[thick_idx][p_idx] = np.mean(np.abs(recon_file.mc_column('absy') - recon_file.column('posy')))
            #Computing energy threshold for classification and relative efficiency
            energy_thr, efficiency = energy_threshold_computation(fitted_model) #contamination set to 2%
            efficiency_on_alpha[thick_idx][p_idx] = efficiency*QE_alpha

            #Constructing the mean cluster size
            mean_cluster_size[thick_idx][p_idx] = np.mean(cluster_size[mask])
            #quantum_efficiency[thick_idx][p_idx] = (n_events-len(cluster_size))/n_events

            #Saving the ratio of 1px wrt all evts if correction is required
            if correct_1px_ratio is True:
                onepx_evts_ratio[thick_idx][p_idx] = len(cluster_size[onepxmask])/len(cluster_size)
            recon_file.close()
    # After having saved the interesting quantities in arrays, analysis is performed.
    #print(quantum_efficiency)
    #constructing the metric for the shift of the mean
    mean_shift_ka = (mean_energy[0]-mu_true_alpha)/mu_true_alpha
    mean_shift_kb = (mean_energy[1]-mu_true_beta)/mu_true_beta
    #constructing the energy resolution
    energy_res_ka = sigmas[0]*2.35 #eV (FWHM)
    energy_res_kb = sigmas[1]*2.35 #eV (FWHM)

    # Plotting the overlapped heatmaps and customizing them.
    fig,ax = double_heatmap(pitch, thickness, mean_shift_ka.flatten(), mean_shift_kb.flatten(), 3)
    plt.title(rf'$\Delta = \frac{{\mu_{{E}}-E_{{K}}}}{{E_{{K}}}}$, $n_{{electronics}}$ = {enc} ENC')
    plt.ylabel(r'Thickness t [$\mu$m]')
    plt.xlabel(r'Pitch p [$\mu$m]')
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

    fig2,ax2 = double_heatmap(pitch, thickness, energy_res_ka.flatten(), energy_res_kb.flatten(), 0)
    plt.title(rf'Energy resolution $\sigma_{{K_{{\alpha, \beta}}}}$ FWHM [eV], $n_{{electronics}}$ = {enc} ENC')
    plt.ylabel(r'Thickness t [$\mu$m]')
    plt.xlabel(r'Pitch p [$\mu$m]')
    # custom yticks. Setting a right yaxis
    twin1 = ax2.twinx()
    twin1.set(ylim=(0, len_yaxis))
    twin1.yaxis.set(ticks=np.arange(0.25, len_yticks/2, 0.5), ticklabels=ticks)
    if save_path is not None:
            plt.savefig(f'{save_path}/energy_res_all_evts.pdf')
    
    heatmap_with_labels(pitch, thickness, efficiency_on_alpha)
    plt.title(rf'Efficiency on $K_{{\alpha}}$ signal, $n_{{electronics}}$ = {enc} ENC')
    plt.xlabel(r'Pitch p [$\mu$m]')
    plt.ylabel(r'Thickness t [$\mu$m]')
    if save_path is not None:
        plt.savefig(f'{save_path}/efficiency.pdf')

    heatmap_with_labels(pitch, thickness, mean_cluster_size, 2)
    plt.title(rf'Average cluster size, $n_{{electronics}}$ = {enc} ENC')
    plt.xlabel(r'Pitch p [$\mu$m]')
    plt.ylabel(r'Thickness t [$\mu$m]')
    if save_path is not None:
        plt.savefig(f'{save_path}/mean_cluster_size.pdf')

    heatmap_with_labels(pitch, thickness, dx, n_digits=7)
    plt.title(rf'Reconstruction shift on x: $\mu(|x_{{MC}} - x_{{recon}}|)$ [cm], $n_{{electronics}}$ = {enc} ENC')
    plt.xlabel(r'Pitch p [$\mu$m]')
    plt.ylabel(r'Thickness t [$\mu$m]')
    if save_path is not None:
        plt.savefig(f'{save_path}/x_shift.pdf')

    heatmap_with_labels(pitch, thickness, dy, n_digits=7)
    plt.title(rf'Reconstruction shift on y: $\mu(|y_{{MC}} - y_{{recon}}|)$ [cm], $n_{{electronics}}$ = {enc} ENC')
    plt.xlabel(r'Pitch p [$\mu$m]')
    plt.ylabel(r'Thickness t [$\mu$m]')
    if save_path is not None:
        plt.savefig(f'{save_path}/y_shift.pdf')
    
    if correct_1px_ratio is True:
        heatmap_with_labels(pitch, thickness, onepx_evts_ratio)
        plt.title(fr'Fraction of events with cluster size = 1, $n_{{electronics}}$ = {enc} ENC')
        plt.xlabel(r'Pitch p [$\mu$m]')
        plt.ylabel(r'Thickness t $\mu$m')
        if save_path is not None:
            plt.savefig(f'{save_path}/fraction_1px_evts.pdf')

    '''
    #Constructing the hist containing the dx and dy
    plt.figure()
    print(f'{enc}, {thick}, {p}')
    values = recon_file.mc_column('absx') - recon_file.column('posx')
    binning = np.linspace(values.min(), values.max(), 100)
    histx = Histogram1d(binning, xlabel='dx').fill(values)
    histx.set_axis_label(0, r'$x_{MC} - x_{recon}$ [cm]')
    histx.plot()

    plt.figure()
    values = recon_file.mc_column('absy') - recon_file.column('posy')
    binning = np.linspace(values.min(), values.max(), 100)
    histx = Histogram1d(binning, xlabel='dy').fill(values)
    histx.set_axis_label(0, r'$y_{MC} - y_{recon}$ [cm]')
    histx.plot()
    
    
    recon_file.close()
    '''


if __name__ == '__main__':
    #Choosing values of enc and thickness from simulated ones.
    enc_ = np.array([20, 30, 40])
    thickness_ = np.array([0.02, 0.025, 0.03, 0.035])*(1e4)
    pitch = 60
    pitch_ = np.array([0.0050, 0.0055, 0.0060])*(1e4)
    enc = 20
    nneighbors = np.arange(1,8)
    zthr = np.linspace(0,1000,50)
    zthr = zthr[zthr<700] 
    #Turning arrays into ints for reading filename correctly
    #thickness_ = thickness_.astype(int)
    #pitch_ = pitch_.astype(int)
    #analyze_grid_thickenc(thickness_, enc_, pitch, **vars(ANALYZE_GRID_ARGPARSER.parse_args()))
    #analyze_grid_thickpitch(thickness_, pitch_, enc, **vars(ANALYZE_GRID_ARGPARSER.parse_args()))
    analyze_grid_zthr(250, 20, 60, zthr, nneighbors)

    plt.show()
