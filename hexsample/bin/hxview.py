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
from hexsample.hist import Histogram1d
from hexsample.io import ReconInputFile
from hexsample.plot import plt


def view(**kwargs):
    """View the file content.
    """
    input_file = ReconInputFile(kwargs['infile'])
    rec_energy = input_file.recon_table.col('energy')
    mc_energy = input_file.mc_table.col('energy')
    cluster_size = input_file.recon_table.col('cluster_size')
    plt.figure('Energy spectrum')
    binning = np.linspace(rec_energy.min(), rec_energy.max(), 100)
    h_rec = Histogram1d(binning).fill(rec_energy)
    h_rec.plot()
    #h_mc = Histogram1d(binning).fill(mc_energy)
    #h_mc.plot()
    plt.figure('Cluster size')
    binning = np.linspace(-0.5, 5.5, 7)
    h = Histogram1d(binning).fill(cluster_size)
    h.plot()
    input_file.close()
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_infile()
    args = parser.parse_args()
    view(**args.__dict__)
