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

"""Event file viewer comparing reconstructed quantities with MC truth.
"""
from ast import literal_eval

import numpy as np

from hexsample.app import ArgumentParser
from hexsample.fileio import ReconInputFile
from hexsample.hist import Histogram1d, Histogram2d
from hexsample.plot import plt, setup_gca
from hexsample.analysis import create_histogram


__description__ = \
"""Simple viewer for comparing reconstructed energy and position with the MC
true values.
"""

# Parser object.
HXVIEW_ARGPARSER = ArgumentParser(description=__description__)
HXVIEW_ARGPARSER.add_infile()

def hxview(**kwargs):
    """View the file content.
    Shows histograms of energy and cluster_size of recon events vs their MC truth.
    """
    input_file = ReconInputFile(kwargs['infile'])
    # Plotting the reconstructed energy and the true energy
    histo = create_histogram(input_file, 'energy', mc = False)
    mc_histo = create_histogram(input_file, 'energy', mc = True)
    plt.figure('Photons energy')
    histo.plot(label='Reconstructed')
    mc_histo.plot(label='MonteCarlo')
    plt.xlabel('Energy [eV]')
    plt.legend()

    # Plotting the reconstructed x and y position and the true position.
    plt.figure('Reconstructed photons position')
    binning = np.linspace(-5. * 0.1, 5. * 0.1, 100)
    x = input_file.column('posx')
    y = input_file.column('posy')
    histo = Histogram2d(binning, binning).fill(x, y)
    histo.plot()
    setup_gca(xlabel='x [cm]', ylabel='y [cm]')
    plt.figure('True photons position')
    x_mc = input_file.mc_column('absx')
    y_mc = input_file.mc_column('absy')
    histo_mc = Histogram2d(binning, binning).fill(x_mc, y_mc)
    histo_mc.plot()
    setup_gca(xlabel='x [cm]', ylabel='y [cm]')
    #Closing the file and showing the figures.
    plt.figure('x-direction resolution')
    binning = np.linspace((x-x_mc).min(), (x-x_mc).max(), 100)
    histx = Histogram1d(binning, xlabel=r'$x - x_{MC}$ [cm]').fill(x-x_mc)
    histx.plot()
    plt.figure('y-direction resolution')
    binning = np.linspace((y-y_mc).min(), (y-y_mc).max(), 100)
    histy = Histogram1d(binning, xlabel=r'$y - y_{MC}$ [cm]').fill(y-y_mc)
    histy.plot()

    input_file.close()
    plt.show()

if __name__ == '__main__':
    hxview(**vars(HXVIEW_ARGPARSER.parse_args()))
