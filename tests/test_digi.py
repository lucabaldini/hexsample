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

"""Test suite for hexsample.digi
"""

import numpy as np

from loguru import logger
import pytest

from hexsample import digi
from hexsample.hexagon import HexagonalLayout
from hexsample.roi import Padding, RegionOfInterest


def test_digi_event_base():
    """Test for the base event class.
    """
    pha = np.full(3, 100.)
    event = digi.DigiEventBase(0, 0, 0, 0, pha)
    print(event)
    print(event.timestamp())

def test_digi_event_sparse():
    """Test for sparse digi event class.
    """
    pha = np.array([50., 150., 25.])
    rows = np.array([1, 2, 3])
    columns = np.array([11, 12, 12])
    event = digi.DigiEventSparse(0, 0, 0, 0, pha, rows, columns)
    print(event)
    print(event.timestamp())
    print(event.ascii())
    # Make sure that the check on the dimensions of the row and column arrays is
    # at work
    with pytest.raises(RuntimeError):
        rows = np.array([1, 2, 3])
        columns = np.array([11, 12, 12, 12])
        event = digi.DigiEventSparse(0, 0, 0, 0, pha, rows, columns)
    with pytest.raises(RuntimeError):
        rows = np.array([1, 2, 3, 4])
        columns = np.array([11, 12, 12])
        event = digi.DigiEventSparse(0, 0, 0, 0, pha, rows, columns)

def test_digitization_sparse(layout: HexagonalLayout = HexagonalLayout.ODD_R,
    num_cols: int = 100, num_rows: int = 100, pitch: float = 0.1, enc: float = 0.,
    gain: float = 0.5, num_pairs: int = 1000, trg_threshold: float = 200.):
    """Test for sparse event digitalization class.
    """
    readout = digi.HexagonalReadoutSparse(layout, num_cols, num_rows, pitch, enc, gain)
    # Pick out a particular pixel...
    col, row = num_cols // 3, num_rows // 4
    col1, row1= num_cols // 6, num_rows // 3
    logger.debug(f'Testing pixel ({col}, {row}) and ({col1}, {row1})...')
    # ... create the x and y arrays of the pair positions in the center of the pixel.
    x0, y0 = readout.pixel_to_world(col, row)
    x1, y1 = readout.pixel_to_world(col1, row1)
    x, y = np.full(int(num_pairs), x0), np.full(int(num_pairs), y0)
    print(len(y))
    x = np.append(x, np.full(int(num_pairs), x1))
    y = np.append(y, np.full(int(num_pairs), y1))
    print(x,y)
    print(len(x), len(y))
    a = readout.read(0., x, y, 100.) #this is a DigiEventSparse
    print(a.ascii())

def test_digi_event(min_col: int = 106, max_col: int = 113, min_row: int = 15,
    max_row: int = 22, padding: Padding = Padding(1, 2, 3, 4)):
    """Build a digi event and make sure it makes sense.
    """
    roi = RegionOfInterest(min_col, max_col, min_row, max_row, padding)
    # The pha is basically the serial readout order, here.
    pha = np.arange(roi.size)
    evt = digi.DigiEvent(0, 0, 0, 0, roi, pha)
    print(evt.ascii())
    i, j = 0, 2
    assert evt.pha[i, j] == 2
    col, row = j + evt.roi.min_col, i + evt.roi.min_row
    assert evt(col, row) == 2

def test_digi_event_comparison():
    """
    """
    padding = Padding(2)
    roi = RegionOfInterest(10, 23, 10, 23, padding)
    pha = np.full(roi.size, 2)
    evt1 = digi.DigiEvent(0, 0, 0, 0, roi, pha)
    evt2 = digi.DigiEvent(0, 0, 0, 0, roi, 1. * pha)
    evt3 = digi.DigiEvent(0, 0, 0, 0, roi, 2. * pha)
    assert evt1 == evt2
    assert evt1 != evt3

def test_digitization(layout: HexagonalLayout = HexagonalLayout.ODD_R, num_cols: int = 100,
    num_rows: int = 100, pitch: float = 0.1, enc: float = 0., gain: float = 0.5,
    num_pairs: int = 1000, trg_threshold: float = 200., padding: Padding = Padding(1)):
    """Create a fake digi event and test all the steps of the digitization.
    """
    readout = digi.HexagonalReadout(layout, num_cols, num_rows, pitch, enc, gain)
    # Pick out a particular pixel...
    col, row = num_cols // 3, num_rows // 4
    logger.debug(f'Testing pixel ({col}, {row})...')
    # ... create the x and y arrays of the pair positions in the center of the pixel.
    x0, y0 = readout.pixel_to_world(col, row)
    x, y = np.full(num_pairs, x0), np.full(num_pairs, y0)
    # Extract the counts: this should provide an array where all the values are
    # zero except the one at position (row, col)---don't forget rows go first in
    # the native numpy representation.
    min_col, min_row, signal = readout.sample(x, y)
    assert signal[row - min_row, col - min_col] == num_pairs
    assert np.nonzero(signal) == (row - min_row, col - min_col)
    # Apply the trigger.
    roi, pha = readout.trigger(signal, trg_threshold, min_col, min_row, padding)
    assert roi.min_col == 2 * (col // 2) - padding.left
    assert roi.max_col == 2 * (col // 2) + 1 + padding.right
    assert roi.min_row == 2 * (row // 2) - padding.bottom
    assert roi.max_row == 2 * (row // 2) + 1 + padding.top
    # And now, redo all the steps and create an actual digi event.
    evt = readout.read(0., x, y, trg_threshold, padding)
    assert evt(col, row) == round(num_pairs * gain)
    print(evt.ascii())
