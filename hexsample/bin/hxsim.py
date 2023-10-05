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

from hexsample import HEXSAMPLE_DATA
from hexsample.app import ArgumentParser
from hexsample.digi import Xpol3
from hexsample.io import DigiOutputFile
from hexsample.mc import PhotonList
from hexsample.roi import Padding
from hexsample.source import LineForest, GaussianBeam, Source
from hexsample.sensor import Material, Sensor


def simulate(**kwargs):
    """Run a simulation.
    """
    spectrum = LineForest(kwargs['srcelement'], kwargs['srclevel'])
    beam = GaussianBeam(kwargs['srcposx'], kwargs['srcposy'], kwargs['srcsigma'])
    source = Source(spectrum, beam)
    material = Material(kwargs['actmedium'], kwargs['fano'])
    sensor = Sensor(material, kwargs['thickness'], kwargs['transdiffsigma'])
    photon_list = PhotonList(source, sensor, kwargs['numevents'])
    readout = Xpol3(kwargs['noise'], kwargs['gain'])
    output_file = DigiOutputFile(kwargs.get('outfile'), mc=True)
    padding = Padding(*kwargs['padding'])
    readout_args = kwargs['trgthreshold'], padding, kwargs['zsupthreshold'], kwargs['offset']
    logger.info('Starting the event loop...')
    for i, mc_event in enumerate(photon_list):
        x, y = mc_event.propagate(sensor.trans_diffusion_sigma)
        digi_event = readout.read(mc_event.timestamp, x, y, *readout_args)
        output_file.add_row(digi_event, mc_event)
    logger.info('Done!')
    output_file.flush()
    output_file.close()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_numevents(1000)
    parser.add_outfile(HEXSAMPLE_DATA / 'hxsim.h5')
    parser.add_source_options()
    parser.add_sensor_options()
    parser.add_readout_options()
    args = parser.parse_args()
    simulate(**args.__dict__)
