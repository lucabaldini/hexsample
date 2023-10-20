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
from hexsample.fitting import fit_histogram
from hexsample.hist import Histogram1d
from hexsample.fileio import ReconInputFile
from hexsample.modeling import Gaussian
from hexsample.plot import plt
from hexsample.analysis import pha_analysis


__description__ = \
"""Simple viewer for reconstructed event lists.
"""

# Parser object.
HXVIEW_ARGPARSER = ArgumentParser(description=__description__)
HXVIEW_ARGPARSER.add_infile()


def hxview(**kwargs):
    """View the file content.
    """
    
    input_file = ReconInputFile(kwargs['infile'])
    h_cluster_size,h_energy_tot = pha_analysis(input_file)
    input_file.close()
    plt.show()
    
    '''
    input_file = ReconInputFile(kwargs['infile'])
    rec_energy = input_file.recon_table.col('energy')
    mc_energy = input_file.mc_table.col('energy')
    cluster_size = input_file.recon_table.col('cluster_size')
    
    plt.figure('Energy spectrum')
    binning = np.linspace(rec_energy.min(), rec_energy.max(), 100)
    h_rec = Histogram1d(binning).fill(rec_energy)
    h_rec.plot()
    model = Gaussian() + Gaussian()
    #model = fit_histogram(GaussianLineForestCuK(), h_rec)
    fit_histogram(model, h_rec, p0=(1., 8000., 150., 1., 8900., 150.))
    model.plot()
    model.stat_box()
    #h_mc = Histogram1d(binning).fill(mc_energy)
    #h_mc.plot()
    plt.figure('Cluster size')
    binning = np.linspace(-0.5, 5.5, 7)
    h = Histogram1d(binning).fill(cluster_size)
    h.plot()
    input_file.close()
    plt.show()
    '''



if __name__ == '__main__':
    hxview(**vars(HXVIEW_ARGPARSER.parse_args()))
