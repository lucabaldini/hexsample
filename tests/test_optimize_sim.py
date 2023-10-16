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

"""Module containing much of the work that went into the optimization of the
simulation, as tracked in https://github.com/lucabaldini/hexsample/issues/12

Strictly speking this is not a unit test, but it's something that's handy to
have floating around for future reference.
"""

import time

import numpy as np

from hexsample import logger
from hexsample import xpol
from hexsample.digi import HexagonalReadout, Xpol3
from hexsample.hexagon import HexagonalLayout
from hexsample.fileio import DigiEvent
from hexsample.mc import PhotonList
from hexsample.sensor import SiliconSensor
from hexsample.source import LineForest, GaussianBeam, Source
from hexsample.roi import RegionOfInterest, Padding



class HexagonalReadoutCompat(HexagonalReadout):

    """Compatibility class implementing the readout behavior up to hexsample 0.3.1,
    i.e., before the simulation optimization described in
    https://github.com/lucabaldini/hexsample/issues/12
    """

    def __init__(self, layout : HexagonalLayout, num_cols : int, num_rows : int,
                 pitch : float, enc : float, gain : float) -> None:
        """Constructor.
        """
        super().__init__(layout, num_cols, num_rows, pitch, enc, gain)
        self._col_binning = np.arange(self.num_cols + 1) - 0.5
        self._row_binning = np.arange(self.num_rows + 1) - 0.5
        self._binning = (self._row_binning, self._col_binning)

    @staticmethod
    def trim_to_roi(array : np.ndarray, roi : RegionOfInterest) -> np.ndarray:
        """Utility function to trim a generic array to a given ROI.

        This is returning the rectangular portion of the input array corresponding
        to the ROI, preserving the original values in that portion.

        Arguments
        ---------
        array : array_like
            The input array.

        roi : RegionOfInterest
            The target region of interest.
        """
        return array[roi.min_row:roi.max_row + 1, roi.min_col:roi.max_col + 1]

    def sample(self, x : np.ndarray, y : np.ndarray) -> np.ndarray:
        """Spatially sample a pair of arrays of x and y coordinates in physical
        space onto logical (hexagonal) coordinates in logical space.

        This is achieved by converting the (x, y) physical coordinates into the
        corresponding (col, row) logical coordinates on the hexagonal grid, and
        then filling a two dimensional histogram in logical space. Note that,
        although the output array represent counts, the corresponding underlying
        dtype is float64, and we do not attempt a cast to integer since the
        very next step in the digitization chain is adding the noise, which by
        its very nature is intrinsically a floating point quantity, even in the
        equivalent noise charge representation.

        Arguments
        ---------
        x : array_like
            The physical x coordinates to sample.

        y : array_like
            The physical y coordinates to sample.
        """
        # pylint: disable=invalid-name
        col, row = self.world_to_pixel(x, y)
        # Note that the histogram takes place in the numpy array representation,
        # that is, rows go first---this way we avoid a transposition to get the
        # array of counts in the proper shape.
        counts, _, _ = np.histogram2d(row, col, self._binning)
        return counts

    def trigger(self, signal : np.ndarray, trg_threshold : float) -> np.ndarray:
        """Apply the trigger to a given signal array and with a fixed threshold.

        Here we downsample in the signal into the 2 x 2 trigger miniclusters,
        we set to zero the content for all the miniclusters below the trigger
        threshold, and we return the zero-suppressed trigger array that can
        be used to calculate the ROI.

        Arguments
        ---------
        signal : array_like
            The num_rows x num_cols array of pixel signals in electron equivalent.

        trg_threshold : float
            The trigger threshold in electron equivalent.
        """
        trg = self.sum_miniclusters(signal)
        self.zero_suppress(trg, trg_threshold)
        self.trigger_id += 1
        return trg

    def calculate_roi(self, trg : np.ndarray, padding : Padding) -> RegionOfInterest:
        """Calculate the region of interest for a given trigger array.

        Arguments
        ---------
        trg : array_like
            The array holding the content of the signal miniclusters.

        padding : Padding
            The padding to be applied to the region of trigger.
        """
        cols = 2 * np.nonzero(trg.sum(axis=0))[0]
        rows = 2 * np.nonzero(trg.sum(axis=1))[0]
        min_col = np.clip(cols.min() - padding.left, 0, self.num_cols)
        max_col = np.clip(cols.max() + 1 + padding.right, 0, self.num_cols)
        min_row = np.clip(rows.min() - padding.top, 0, self.num_rows)
        max_row = np.clip(rows.max() + 1 + padding.bottom, 0, self.num_rows)
        return RegionOfInterest(min_col, max_col, min_row, max_row, padding)

    def digitize(self, signal : np.ndarray, roi : RegionOfInterest,
        zero_sup_threshold : int = 0, offset : int = 0) -> np.ndarray:
        """Digitize the actual signal within a given ROI.

        Arguments
        ---------
        signal : array_like
            The input array of pixel signals to be digitized.

        roi : RegionOfInterest
            The target ROI.

        zero_sup_threshold : int
            Zero-suppression threshold in ADC counts.

        offset : int
            Optional offset in ADC counts to be applied before the zero suppression.
        """
        # Trim the signal to the given ROI...
        pha = self.trim_to_roi(signal, roi)
        # ... add the noise.
        if self.enc > 0:
            pha += np.random.normal(0., self.enc, size=pha.shape)
        # ... apply the conversion between electrons and ADC counts...
        pha *= self.gain
        # ... round to the neirest integer...
        pha = np.round(pha).astype(int)
        # ... if necessary, add the offset for diagnostic events...
        pha += offset
        # ... zero suppress the thing...
        self.zero_suppress(pha, zero_sup_threshold)
        # ... flatten the array to simulate the serial readout and return the
        # array as the BEE would have.
        return pha.flatten()

    def read(self, timestamp : float, x : np.ndarray, y : np.ndarray, trg_threshold : float,
        padding : Padding, zero_sup_threshold : int = 0, offset : int = 0) -> DigiEvent:
        """Readout an event.

        Arguments
        ---------
        timestamp : float
            The event timestamp.

        x : array_like
            The physical x coordinates of the input charge.

        y : array_like
            The physical x coordinates of the input charge.

        trg_threshold : float
            Trigger threshold in electron equivalent.

        padding : Padding
            The padding to be applied to the ROT.

        zero_sup_threshold : int
            Zero suppression threshold in ADC counts.

        offset : int
            Optional offset in ADC counts to be applied before the zero suppression.
        """
        # pylint: disable=invalid-name, too-many-arguments
        signal = self.sample(x, y)
        trg = self.trigger(signal, trg_threshold)
        roi = self.calculate_roi(trg, padding)
        pha = self.digitize(signal, roi, zero_sup_threshold, offset)
        seconds, microseconds, livetime = self.latch_timestamp(timestamp)
        return DigiEvent(self.trigger_id, seconds, microseconds, livetime, roi, pha)


# XPOL-III like readouts, with the old and the new, streamlined implementetion.
# Note that we set the noise to 0. in order to allow for a deterministic
# comparison among the two readouts.
OLD_READOUT = HexagonalReadoutCompat(xpol.XPOL1_LAYOUT, *xpol.XPOL3_SIZE, xpol.XPOL_PITCH, 0., 1.)
NEW_READOUT = HexagonalReadout(xpol.XPOL1_LAYOUT, *xpol.XPOL3_SIZE, xpol.XPOL_PITCH, 0., 1.)


def _compare_readouts(x, y, trg_threshold=200., padding=Padding(2)):
    """
    """
    args = 0., x, y, trg_threshold, padding
    old = OLD_READOUT.read(*args)
    #print(old.ascii())
    new = NEW_READOUT.read(*args)
    #print(new.ascii())
    assert np.allclose(old.pha, new.pha)
    assert old.roi.min_col == new.roi.min_col
    assert old.roi.max_col == new.roi.max_col
    assert old.roi.min_row == new.roi.min_row
    assert old.roi.max_row == new.roi.max_row
    return old, new

def _test_pixel_centers():
    """Shoot a few toy events at the center of four selected pixels (with all
    possible parities) and make sure that the new readout matches the old one
    exactly.
    """
    for col, row in ((150, 150), (151, 150), (150, 151), (151, 151)):
        x0, y0 = OLD_READOUT.pixel_to_world(col, row)
        logger.info(f'Generating @ ({col}, {row}) -> ({x0}, {y0})')
        x = np.full(2500, x0)
        y = np.full(2500, y0)
        old, new = _compare_readouts(x, y)

def test_photon_list(num_photons=1000):
    """Realistic comparison with a sensible photon list.
    """
    spectrum = LineForest('Cu', 'K')
    beam = GaussianBeam(0., 0., 0.1)
    source = Source(spectrum, beam)
    sensor = SiliconSensor(0.05, 40.)
    photon_list = PhotonList(source, sensor, num_photons)
    for mc_event in photon_list:
        x, y = mc_event.propagate(sensor.trans_diffusion_sigma)
        old, new = _compare_readouts(x, y)

def test_timing(sigma=0.0006, num_pairs=2250, num_photons=10000):
    """Time the sampling routine.
    """
    x = np.random.normal(0., sigma, size=num_pairs)
    y = np.random.normal(0., sigma, size=num_pairs)
    padding = Padding(2)
    trg_threshold = 300.
    #
    logger.info('Timing world_to_pixel()...')
    start_time = time.time()
    for i in range(num_photons):
        col, row = NEW_READOUT.world_to_pixel(x, y)
    elapsed_time = time.time() - start_time
    evt_us = 1.e6 * elapsed_time / num_photons
    logger.info(f'Elapsed time: {elapsed_time:.3f} s, {evt_us:.1f} us per event.')
    #
    logger.info('Timing sample()...')
    start_time = time.time()
    for i in range(num_photons):
        min_col, min_row, signal = NEW_READOUT.sample(x, y)
    elapsed_time = time.time() - start_time
    evt_us = 1.e6 * elapsed_time / num_photons
    logger.info(f'Elapsed time: {elapsed_time:.3f} s, {evt_us:.1f} us per event.')
    #
    logger.info('Timing trigger()...')
    start_time = time.time()
    for i in range(num_photons):
        roi, pha = NEW_READOUT.trigger(signal, trg_threshold, min_col, min_row, padding)
    elapsed_time = time.time() - start_time
    evt_us = 1.e6 * elapsed_time / num_photons
    logger.info(f'Elapsed time: {elapsed_time:.3f} s, {evt_us:.1f} us per event.')
