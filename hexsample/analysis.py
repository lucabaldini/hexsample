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

import numpy as np

from hexsample.fitting import fit_histogram
from hexsample.hist import Histogram1d
from hexsample.io import ReconInputFile
from hexsample.modeling import Gaussian
from hexsample.plot import plt, setup_gca


def cluster_size_analysis(input_file : ReconInputFile):
    """
    """
    clu_size = input_file.recon_column('cluster_size')

def pha_analysis(input_file : ReconInputFile):
    """
    """
    rec_energy = input_file.recon_column('energy')
    clu_size = input_file.recon_column('cluster_size')

    plt.figure('Cluster size')
    h = Histogram1d(np.linspace(-0.5, 5.5, 7)).fill(clu_size)
    h.plot()

    mask = clu_size <= 1
    frac = mask.sum() / len(mask)
    print(f'Selection efficiency = {frac}')
    rec_energy = rec_energy[mask]

    plt.figure('Energy')
    binning = np.linspace(rec_energy.min(), rec_energy.max(), 100)
    h = Histogram1d(binning).fill(rec_energy)
    h.plot()
    setup_gca(xlabel='Energy [keV]')
    model = Gaussian() + Gaussian()
    fit_histogram(model, h, p0=(1., 8050., 200., 0.5, 8900., 100.), xmin=7890)
    model.plot()
    model.stat_box()



if __name__ == '__main__':
    thickness = 500
    enc = 40
    thr = 2 * enc
    file_path = f'/home/lbaldini/hexsampledata/sim_thick{thickness}_enc{enc}_diffx1_recon_thr{thr}.h5'
    recon_file = ReconInputFile(file_path)
    #cluster_size_analysis(recon_file)
    pha_analysis(recon_file)
    plt.show()
