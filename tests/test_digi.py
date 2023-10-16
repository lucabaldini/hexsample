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

from hexsample.digi import DigiEvent, HexagonalReadout
from hexsample.hexagon import HexagonalLayout
from hexsample.roi import Padding, RegionOfInterest

def test_digi_event(min_col : int = 106, max_col : int = 113, min_row : int = 15,
    max_row : int = 22, padding : Padding = Padding(1, 2, 3, 4)):
    """Build a digi event and make sure it makes sense.
    """
    roi = RegionOfInterest(min_col, max_col, min_row, max_row, padding)
    # The pha is basically the serial readout order, here.
    pha = np.arange(roi.size)
    evt = DigiEvent(0, 0, 0, 0, roi, pha)
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
    evt1 = DigiEvent(0, 0, 0, 0, roi, pha)
    evt2 = DigiEvent(0, 0, 0, 0, roi, 1. * pha)
    evt3 = DigiEvent(0, 0, 0, 0, roi, 2. * pha)
    assert evt1 == evt2
    assert evt1 != evt3

def test_digitization(layout : HexagonalLayout = HexagonalLayout.ODD_R, num_cols : int = 100,
    num_rows : int = 100, pitch : float = 0.1, enc : float = 0., gain : float = 0.5,
    num_pairs : int = 1000, trg_threshold : float = 200., padding : Padding = Padding(1)):
    """Create a fake digi event and test all the steps of the digitization.
    """
    readout = HexagonalReadout(layout, num_cols, num_rows, pitch, enc, gain)
    # Pick out a particular pixel...
    col, row = num_cols // 3, num_rows // 4
    logger.debug(f'Testing pixel ({col}, {row})...')
    # ... create the x and y arrays of the pair positions in the center of the pixel.
    x0, y0 = readout.pixel_to_world(col, row)
    x, y = np.full(num_pairs, x0), np.full(num_pairs, y0)
    # Extract the counts: this should provide an array where all the values are
    # zero except the one at position (row, col)---don't forget rows go first in
    # the native numpy representation.
    signal = readout.sample(x, y)
    assert signal[row, col] == num_pairs
    assert np.nonzero(signal) == (row, col)
    # Apply the trigger.
    trg = readout.trigger(signal, trg_threshold)
    assert trg[row // 2, col // 2] == num_pairs
    assert np.nonzero(trg) == (row // 2, col // 2)
    # Calculate the ROI.
    roi = readout.calculate_roi(trg, padding)
    assert roi.min_col == 2 * (col // 2) - padding.left
    assert roi.max_col == 2 * (col // 2) + 1 + padding.right
    assert roi.min_row == 2 * (row // 2) - padding.bottom
    assert roi.max_row == 2 * (row // 2) + 1 + padding.top
    # And now, redo all the steps and create an actual digi event.
    evt = readout.read(0., x, y, trg_threshold, padding)
    assert evt(col, row) == round(num_pairs * gain)
    print(evt.ascii())
