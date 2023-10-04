# Copyright (C) 2023 luca.baldini@pi.infn.it
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Test suite for hexsample.io
"""

from loguru import logger
import numpy as np

from hexsample import HEXSAMPLE_DATA
from hexsample.digi import DigiEvent
from hexsample.io import DigiInputFile, DigiOutputFile
from hexsample.mc import MonteCarloEvent
from hexsample.roi import RegionOfInterest, Padding



def _mc_event(index : int) -> MonteCarloEvent:
    """Create a bogus MonteCarloEvent object with index-dependent properties.
    """
    return MonteCarloEvent(0.1 * index, 5.9, 0., 0., 0.02, 1000 + index)

def _digi_event(index : int) -> DigiEvent:
    """Create a bogus DigiEvent object with index-dependent properties.
    """
    roi = RegionOfInterest(100, 107, 150, 155 + index * 2, Padding(2))
    pha = np.full(roi.size, index)
    return DigiEvent(index, index, index, 0, roi, pha)

def _test_write(file_path, num_events : int = 10):
    """Small test writing a bunch of toy event strcutures to file.
    """
    output_file = DigiOutputFile(file_path, mc=True)
    for i in range(num_events):
        output_file.add_row(_digi_event(i), _mc_event(i))
    output_file.close()

def _test_read(file_path):
    """Small test interating over an input file.
    """
    input_file = DigiInputFile(file_path)
    print(input_file)
    for i, event in enumerate(input_file):
        print(event.ascii())
        print(input_file.mc_event(i))
        target = _digi_event(i)
        assert event.trigger_id == target.trigger_id
        assert event.seconds == target.seconds
        assert event.microseconds == target.microseconds
        assert event.roi == target.roi
        assert (event.pha == target.pha).all()
    input_file.close()

def test():
    """Write and read back a simple digi file.
    """
    file_path = HEXSAMPLE_DATA / 'test_io.h5'
    logger.info(f'Testing output file {file_path}...')
    _test_write(file_path)
    logger.info(f'Testing input file {file_path}...')
    _test_read(file_path)
