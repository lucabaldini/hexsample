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

"""Analysis facilities.
"""

from __future__ import annotations

import numpy as np

from typing import Optional, Tuple

from hexsample.fitting import fit_histogram
from hexsample.hist import Histogram1d
from hexsample.fileio import ReconInputFile
from hexsample.modeling import Gaussian
from hexsample.plot import plt, setup_gca



def absz_analysis(input_file : ReconInputFile):
    """
    """
    absz = input_file.mc_column('absz')
    h = Histogram1d(np.linspace(0., 0.06, 100)).fill(absz)
    h.plot()
    setup_gca(logy=True)


def cluster_size_analysis(input_file : ReconInputFile):
    """
    """
    clu_size = input_file.recon_column('cluster_size')

def pha_analysis(input_file : ReconInputFile, PHA_cut_value : float=0) -> Tuple[Histogram1d]:
    """
        This function returns two Histogram1d of energy and cluster_size for a 
        reconstructed file. 
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
    model = Gaussian() + Gaussian()
    fit_histogram(model, h_energy_tot, p0=(1., 8000., 150., 1., 8900., 150.))
    #model = fit_histogram(Gaussian(), h_energy, p0=(1., 8900., 100.), xmin=8600)
    model.plot()
    model.stat_box()
    #Computing the selection efficiency
    mask = clu_size <= 1 #those are the events that did not trigger the chip or that triggered just one pixel
    frac = mask.sum() / len(mask)
    print(f'Selection efficiency = {frac}')
    rec_energy = rec_energy[mask] #Recorded energy for single-pixel events
    '''
    #Creating energy plot for single-pixel triggered events
    plt.figure('Energy of single-pixel triggered events')
    binning_energy = np.linspace(rec_energy.min(), rec_energy.max(), 100)
    h_energy_triggered = Histogram1d(binning_energy).fill(rec_energy)
    h_energy_triggered.plot()
    setup_gca(xlabel='Energy [keV]')
    '''
    


    return h_clu_size,h_energy_tot



if __name__ == '__main__':
    thickness = 500
    enc = 10
    thr = 2 * enc
    file_path = f'/home/lbaldini/hexsampledata/sim_{thickness}um_{enc}enc_recon_nn2_thr{thr}.h5'
    recon_file = ReconInputFile(file_path)
    #cluster_size_analysis(recon_file)
    pha_analysis(recon_file)
    #absz_analysis(recon_file)
    plt.show()
