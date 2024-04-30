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

from hexsample import logger
from hexsample.app import ArgumentParser, check_required_args
from hexsample.clustering import ClusteringNN
from hexsample.readout import HexagonalReadoutMode, HexagonalReadoutSparse, HexagonalReadoutRectangular, HexagonalReadoutCircular
from hexsample.fileio import DigiInputFileBase, DigiInputFileSparse, DigiInputFileRectangular, DigiInputFileCircular, DigiInputFile, ReconOutputFile, peek_readout_type
from hexsample.hexagon import HexagonalLayout
from hexsample.recon import ReconEvent


__description__ = \
"""Run the reconstruction on a file produced by hxsim.py
"""

# Parser object.
HXRECON_ARGPARSER = ArgumentParser(description=__description__)
HXRECON_ARGPARSER.add_infile()
HXRECON_ARGPARSER.add_suffix('recon')
HXRECON_ARGPARSER.add_clustering_options()


def hxrecon(**kwargs):
    """Application main entry point.
    """
    check_required_args(hxrecon, 'infile', **kwargs)
    # Note we cast the input file to string, in case it happens to be a pathlib.Path object.
    input_file_path = str(kwargs['infile'])
    if not input_file_path.endswith('.h5'):
        raise RuntimeError('Input file {input_file_path} does not look like a HDF5 file')
    
    # It is necessary to extract the reaodut type because every readout type
    # corresponds to a different DigiEvent type.
    # If there is no info about the readout, we try to reconstruct with a Rectangular
    # readout mode, that is the mode for all the old reconstruction before the 
    # distinction between different readouts was implemented.
    try:
        readout_mode = peek_readout_type(input_file_path)
        # Now we can construct a set of if/else.
        if readout_mode is HexagonalReadoutMode.SPARSE:
            input_file = DigiInputFileSparse(input_file_path)
            header = input_file.header
            args = HexagonalLayout(header['layout']), header['numcolumns'], header['numrows'],\
                header['pitch'], header['noise'], header['gain']
            readout = HexagonalReadoutSparse(*args)
            logger.info(f'Readout chip: {readout}')
        elif readout_mode is HexagonalReadoutMode.RECTANGULAR:
            input_file = DigiInputFile(input_file_path)
            header = input_file.header
            args = HexagonalLayout(header['layout']), header['numcolumns'], header['numrows'],\
                header['pitch'], header['noise'], header['gain']
            readout = HexagonalReadoutRectangular(*args)
            logger.info(f'Readout chip: {readout}')
        elif readout_mode is HexagonalReadoutMode.CIRCULAR:
            input_file = DigiInputFileCircular(input_file_path)
            header = input_file.header
            args = HexagonalLayout(header['layout']), header['numcolumns'], header['numrows'],\
                header['pitch'], header['noise'], header['gain']
            readout = HexagonalReadoutCircular(*args)
            logger.info(f'Readout chip: {readout}')
    except RuntimeError:
        input_file = DigiInputFile(input_file_path)
        header = input_file.header
        args = HexagonalLayout(header['layout']), header['numcolumns'], header['numrows'],\
            header['pitch'], header['noise'], header['gain']
        #readout = HexagonalReadoutCircular(*args)
        readout = HexagonalReadoutRectangular(*args)
        logger.info(f'Readout chip: {readout}')
    # When the readout tipology is determined, the event is clustered ...   
    clustering = ClusteringNN(readout, kwargs['zsupthreshold'], kwargs['nneighbors'])
    suffix = kwargs['suffix']
    output_file_path = input_file_path.replace('.h5', f'_{suffix}.h5')
    # ... and saved into an output file.
    output_file = ReconOutputFile(output_file_path)
    output_file.update_header(**kwargs)
    output_file.update_digi_header(**input_file.header)
    for i, event in tqdm(enumerate(input_file)):
        cluster = clustering.run(event)
        args = event.trigger_id, event.timestamp(), event.livetime, cluster
        recon_event = ReconEvent(*args)
        mc_event = input_file.mc_event(i)
        output_file.add_row(recon_event, mc_event)
    output_file.flush()
    input_file.close()
    output_file.close()
    return output_file_path



if __name__ == '__main__':
    hxrecon(**vars(HXRECON_ARGPARSER.parse_args()))
