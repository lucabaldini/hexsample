#!/usr/bin/env python
#
# Copyright (C) 2022--2023 luca.baldini@pi.infn.it
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

"""Simple simulation.
"""

from loguru import logger
import numpy as np
from tqdm import tqdm

from hexsample import rng
from hexsample import HEXSAMPLE_DATA
from hexsample.app import ArgumentParser
from hexsample.readout import HexagonalReadoutMode, readout_chip
from hexsample.fileio import DigiDescriptionSparse, DigiDescriptionRectangular,\
    DigiDescriptionCircular, digioutput_class
from hexsample.hexagon import HexagonalLayout
from hexsample.mc import PhotonList
from hexsample.roi import Padding
from hexsample.source import LineForest, GaussianBeam, Source, UniformBeam
from hexsample.sensor import Material, Sensor


__description__ = \
"""Simulate a list of digitized events from an arbitrary X-ray source.
"""

# Parser object.
HXSIM_ARGPARSER = ArgumentParser(description=__description__)
HXSIM_ARGPARSER.add_numevents(1000)
HXSIM_ARGPARSER.add_outfile(HEXSAMPLE_DATA / 'hxsim.h5')
HXSIM_ARGPARSER.add_seed()
HXSIM_ARGPARSER.add_source_options()
HXSIM_ARGPARSER.add_sensor_options()
HXSIM_ARGPARSER.add_readout_options()


def hxsim(**kwargs):
    """Application main entry point.
    """
    # pylint: disable=too-many-locals, invalid-name
    #Initializing chip features: layout and readout
    args = HexagonalLayout(kwargs['layout']), kwargs['numcolumns'], kwargs['numrows'],\
        kwargs['pitch'], kwargs['noise'], kwargs['gain']
    readout_mode = HexagonalReadoutMode(kwargs['readoutmode'])
    # Is there any nicer way to do this? See https://github.com/lucabaldini/hexsample/issues/51
    if readout_mode is HexagonalReadoutMode.SPARSE:
        readout_args = kwargs['trgthreshold'], kwargs['zsupthreshold'], kwargs['offset']
    elif readout_mode is HexagonalReadoutMode.RECTANGULAR:
        padding = Padding(*kwargs['padding'])
        readout_args = kwargs['trgthreshold'], padding, kwargs['zsupthreshold'], kwargs['offset']
    elif readout_mode is HexagonalReadoutMode.CIRCULAR:
        readout_args = kwargs['trgthreshold'], kwargs['zsupthreshold'], kwargs['offset']
    else:
        raise RuntimeError
    readout = readout_chip(readout_mode, *args)
    logger.info(f'Readout chip: {readout}')
    #Initializing the active medium features
    material = Material(kwargs['actmedium'], kwargs['fano'])
    sensor = Sensor(material, kwargs['thickness'], kwargs['transdiffsigma'])
    #Initializing the X-ray source
    rng.initialize(seed=kwargs['seed'])
    spectrum = LineForest(kwargs['srcelement'], kwargs['srclevel'])
    if kwargs['beamshape'] == 'uniform':
        #logger.info(f'Beam shape: {kwargs['beamshape']}')
        # Readout has all the info necessary for initializing the source correctly,
        # in particular for the uniform beam, it is necessary to delimit the active
        # area of the detector.
        x_min, y_min = readout.pixel_to_world(0, kwargs['numrows'])
        x_max, y_max = readout.pixel_to_world(kwargs['numcolumns'], 0)
        beam = UniformBeam(x_min, x_max, y_min, y_max)
    else:
        #logger.info(f'Beam shape: {kwargs['beamshape']}')
        beam = GaussianBeam(kwargs['srcposx'], kwargs['srcposy'], kwargs['srcsigma'])
    source = Source(spectrum, beam)
    photon_list = PhotonList(source, sensor, kwargs['numevents'])
    output_file_path = kwargs.get('outfile')
    output_file = digioutput_class(readout_mode)(output_file_path)
    output_file.update_header(**kwargs)
    logger.info('Starting the event loop...')
    for mc_event in tqdm(photon_list):
        x, y = mc_event.propagate(sensor.trans_diffusion_sigma)
        digi_event = readout.read(mc_event.timestamp, x, y, *readout_args)
        output_file.add_row(digi_event, mc_event)
    logger.info('Done!')
    output_file.flush()
    output_file.close()
    return output_file_path



if __name__ == '__main__':
    hxsim(**vars(HXSIM_ARGPARSER.parse_args()))
