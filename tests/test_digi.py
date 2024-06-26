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

from hexsample.digi import DigiEventBase, DigiEventSparse, DigiEventRectangular, DigiEventCircular
from hexsample.hexagon import HexagonalGrid, HexagonalLayout
from hexsample.readout import HexagonalReadoutRectangular, HexagonalReadoutSparse, HexagonalReadoutCircular
from hexsample.roi import Padding, RegionOfInterest


def test_digi_event_base():
    """Test for the base event class.
    """
    pha = np.full(3, 100.)
    event = DigiEventBase(0, 0, 0, 0, pha)
    print(event)
    print(event.timestamp())

def test_digi_event_sparse():
    """Test for sparse digi event class.
    """
    pha = np.array([50., 150., 25.])
    rows = np.array([1, 2, 3])
    columns = np.array([11, 12, 12])
    event = DigiEventSparse(0, 0, 0, 0, pha, rows, columns)
    print(event)
    #print(event.highest_pixel())
    print(event.timestamp())
    print(event.ascii())
    # Make sure that the check on the dimensions of the row and column arrays is
    # at work
    with pytest.raises(RuntimeError):
        rows = np.array([1, 2, 3])
        columns = np.array([11, 12, 12, 12])
        event = DigiEventSparse(0, 0, 0, 0, pha, rows, columns)
    with pytest.raises(RuntimeError):
        rows = np.array([1, 2, 3, 4])
        columns = np.array([11, 12, 12])
        event = DigiEventSparse(0, 0, 0, 0, pha, rows, columns)

def test_digitization_sparse(layout: HexagonalLayout = HexagonalLayout.ODD_R,
    num_cols: int = 100, num_rows: int = 100, pitch: float = 0.1, enc: float = 0.,
    gain: float = 0.5, num_pairs: int = 1000, trg_threshold: float = 200.):
    """Test for sparse event digitalization class.
    """
    readout = HexagonalReadoutSparse(layout, num_cols, num_rows, pitch, enc, gain)
    # Pick out some particular pixels...
    col1, row1 = num_cols // 3, num_rows // 4
    col2, row2 = col1 + 8, row1 + 5
    col3, row3 = col1 + 4, row1 + 2
    logger.debug(f'Testing pixel ({col1}, {row1}) and ({col2}, {row2})...')
    # ... create the x and y arrays of the pair positions in the center of the pixel.
    x1, y1 = readout.pixel_to_world(col1, row1)
    x2, y2 = readout.pixel_to_world(col2, row2)
    x, y = np.full(int(num_pairs), x1), np.full(int(num_pairs), y1)
    x = np.append(x, np.full(num_pairs, x2))
    y = np.append(y, np.full(num_pairs, y2))
    # Add one more pixel below the trigger threshold, that we want to see disappear
    # in the final event.
    logger.debug(f'Adding pixel ({col3}, {row3}) below teh trigger threshold...')
    x3, y3 = readout.pixel_to_world(col3, row3)
    n = int(0.5 * trg_threshold)
    x = np.append(x, np.full(n, x3))
    y = np.append(y, np.full(n, y3))
    event = readout.read(0., x, y, 100.) #this is a DigiEventSparse
    print(event.ascii())

#@pytest.mark.skip('Under development')
def test_digi_event_circular():
    """Test for circular digi event class.
    """
    pha = np.array([550., 15., 0., 5., 72., 88, 100])
    row = 1
    column = 11
    # Defining the hexagonal grid
    grid = HexagonalGrid(HexagonalLayout.ODD_R, 100, 100, 0.1)
    event = DigiEventCircular(0, 0, 0, 0, pha, row, column)
    print(event)
    print(event.timestamp())
    #print(event.ascii())
    # Make sure that the check on the dimensions of the row and column arrays is
    # at work

#@pytest.mark.skip('Under development')
def test_digitization_circular(layout: HexagonalLayout = HexagonalLayout.ODD_R,
    num_cols: int = 100, num_rows: int = 100, pitch: float = 0.1, enc: float = 0.,
    gain: float = 0.5, num_pairs: int = 1000, trg_threshold: float = 200.):
    """Test for circular event digitalization class.
    """
    readout = HexagonalReadoutCircular(layout, num_cols, num_rows, pitch, enc, gain)
    # Pick out some particular pixels, we expect only the one with higher PHA
    # to be saved in the DigiEventCircular.
    col1, row1 = 2, 4
    col2, row2 = col1 + 8, row1 + 5
    logger.debug(f'Testing pixel ({col1}, {row1}) and ({col2}, {row2})...')
    # ... create the x and y arrays of the pair positions in the center of the pixel.
    x1, y1 = readout.pixel_to_world(col1, row1)
    x2, y2 = readout.pixel_to_world(col2, row2)
    x, y = np.full(int(num_pairs), x1), np.full(int(num_pairs), y1)
    x = np.append(x, np.full(num_pairs, x2))
    y = np.append(y, np.full(num_pairs, y2))
    event = readout.read(0., x, y, 100.) #this is a DigiEventCircular
    print(event.ascii())

def test_digi_event_rectangular(min_col: int = 106, max_col: int = 113, min_row: int = 15,
    max_row: int = 22, padding: Padding = Padding(1, 2, 3, 4)):
    """Build a digi event and make sure it makes sense.
    """
    roi = RegionOfInterest(min_col, max_col, min_row, max_row, padding)
    # The pha is basically the serial readout order, here.
    pha = np.arange(roi.size)
    evt = DigiEventRectangular(0, 0, 0, 0, pha, roi)
    print(evt.highest_pixel())
    print(evt.ascii())
    i, j = 0, 2
    assert evt.pha[i, j] == 2
    col, row = j + evt.roi.min_col, i + evt.roi.min_row
    assert evt(col, row) == 2

def test_digi_event_rectangular_comparison():
    """
    """
    padding = Padding(2)
    roi = RegionOfInterest(10, 23, 10, 23, padding)
    pha = np.full(roi.size, 2)
    evt1 = DigiEventRectangular(0, 0, 0, 0, pha, roi)
    evt2 = DigiEventRectangular(0, 0, 0, 0, 1. * pha, roi)
    evt3 = DigiEventRectangular(0, 0, 0, 0, 2. * pha, roi)
    assert evt1 == evt2
    assert evt1 != evt3

def test_digitization(layout: HexagonalLayout = HexagonalLayout.ODD_R, num_cols: int = 100,
    num_rows: int = 100, pitch: float = 0.1, enc: float = 0., gain: float = 0.5,
    num_pairs: int = 1000, trg_threshold: float = 200., padding: Padding = Padding(1)):
    """Create a fake digi event and test all the steps of the digitization.
    """
    readout = HexagonalReadoutRectangular(layout, num_cols, num_rows, pitch, enc, gain)
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
