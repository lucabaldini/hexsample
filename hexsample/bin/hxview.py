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
from hexsample.modeling import Gaussian
from hexsample.plot import plt
from hexsample.analysis import create_histogram, fit_histogram


__description__ = \
"""Simple viewer for reconstructed event lists.
"""

# Parser object.
HXVIEW_ARGPARSER = ArgumentParser(description=__description__)
HXVIEW_ARGPARSER.add_infile()
HXVIEW_ARGPARSER.add_argument("attribute", type=str, help='Attribute to be viewed.\
                               To be taken from\ the following list: \
                               trigger_id (int),  timestamp (float),  livetime (int)  \
                               roi_size (int), energy (float), position (Tuple[float,float])\
                               cluster_size (int), roi_size (int)')
HXVIEW_ARGPARSER.add_argument("mc_table", type=bool, help='Tells if the quantities are in mc table')


def hxview(**kwargs):
    """View the file content.
       Shows histograms of energy and cluster_size. 
    """
    input_file = ReconInputFile(kwargs['infile'])
    attribute = kwargs['attribute']
    is_mc = kwargs['mc_table']
    histo = create_histogram(input_file, attribute, mc = is_mc)
    plt.figure()
    histo.plot()
    plt.xlabel(attribute)
    input_file.close()
    plt.show()

    


if __name__ == '__main__':
    hxview(**vars(HXVIEW_ARGPARSER.parse_args()))
