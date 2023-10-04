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
from hexsample.sensor import SiliconSensor


def simulate(**kwargs):
    """Run a simulation.
    """
    output_file_path = kwargs.get('outfile')
    num_events = kwargs.get('numevents')
    thickness = kwargs.get('thickness', 0.030)
    trg_threshold = kwargs.get('trgthreshold')
    zero_sup_threshold = kwargs.get('zsupthreshold')
    padding = Padding(2)
    spectrum = LineForest('Cu', 'K')
    beam = GaussianBeam()
    source = Source(spectrum, beam)
    sensor = SiliconSensor(thickness)
    photon_list = PhotonList(source, sensor, num_events)
    readout = Xpol3()
    output_file = DigiOutputFile(output_file_path, mc=True)
    logger.info('Starting the event loop...')
    for i, mc_event in enumerate(photon_list):
        x, y = mc_event.propagate(sensor.trans_diffusion_sigma)
        digi_event = readout.read(mc_event.timestamp, x, y, trg_threshold, padding)
        output_file.add_row(digi_event, mc_event)
    logger.info('Done!')
    output_file.flush()
    output_file.close()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_outfile(HEXSAMPLE_DATA / 'hxsim.h5')
    parser.add_numevents(1000)
    parser.add_trgthreshold()
    parser.add_zsupthreshold()
    args = parser.parse_args()
    simulate(**args.__dict__)
