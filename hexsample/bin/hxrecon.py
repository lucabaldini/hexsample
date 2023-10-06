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

"""Event reconstruction.
"""

from tqdm import tqdm

from hexsample.app import ArgumentParser
from hexsample.clustering import ClusteringNN
from hexsample.digi import Xpol3
from hexsample.io import DigiInputFile, ReconOutputFile
from hexsample.recon import ReconEvent


__description__ = \
"""Run the reconstruction on a file produced by hxsim.py
"""

# Parser object.
HXRECON_ARGPARSER = ArgumentParser(description=__description__)
HXRECON_ARGPARSER.add_infile()
HXRECON_ARGPARSER.add_clustering_options()


def hxrecon(**kwargs):
    """Application main entry point.
    """
    file_path = kwargs['infile']
    clustering = ClusteringNN(Xpol3(), kwargs['zsupthreshold'], kwargs['nneighbors'])
    input_file = DigiInputFile(file_path)
    output_file_path = file_path.replace('.h5', '_recon.h5')
    output_file = ReconOutputFile(output_file_path, mc=True)
    for i, event in tqdm(enumerate(input_file)):
        cluster = clustering.run(event)
        args = event.trigger_id, event.timestamp(), event.livetime, event.roi.size, cluster
        recon_event = ReconEvent(*args)
        mc_event = input_file.mc_event(i)
        output_file.add_row(recon_event, mc_event)
    output_file.flush()
    input_file.close()
    output_file.close()
    return output_file_path



if __name__ == '__main__':
    hxrecon(**HXRECON_ARGPARSER.parse_args().__dict__)
