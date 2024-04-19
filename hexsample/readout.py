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

"""Readout facilities facilities.
"""

from collections import Counter
from typing import Tuple

import numpy as np

from hexsample import rng
from hexsample.digi import DigiEventSparse, DigiEventRectangular, DigiEventCircular
from hexsample.hexagon import HexagonalGrid, HexagonalLayout
from hexsample.roi import Padding, RegionOfInterest
from hexsample import xpol



class HexagonalReadoutBase(HexagonalGrid):

    """Description of a pixel readout chip on a hexagonal matrix.

    Arguments
    ---------
    layout : HexagonalLayout
        The layout of the hexagonal matrix.

    num_cols : int
        The number of columns in the readout.

    num_rows : int
        The number of rows in the readout.

    pitch : float
        The readout pitch in cm.

    enc : float
        The equivalent noise charge in electrons.

    gain : float
        The readout gain in ADC counts per electron.
    """

    def __init__(self, layout: HexagonalLayout, num_cols: int, num_rows: int,
                 pitch: float, enc: float, gain: float) -> None:
        """Constructor.
        """
        # pylint: disable=too-many-arguments
        super().__init__(layout, num_cols, num_rows, pitch)
        self.enc = enc
        self.gain = gain
        self.shape = (self.num_rows, self.num_cols)
        self.trigger_id = -1

    @staticmethod
    def discriminate(array: np.ndarray, threshold: float) -> np.ndarray:
        """Utility function acting as a simple constant-threshold discriminator
        over a generic array. This returns a boolean mask with True for all the
        array elements larger than the threshold.

        This is intented to avoid possible confusion between strict and loose
        comparison operators (e.g., < vs <=) when comparing the content of an array
        with a threshold, and all functions downstream doing this (e.g., zero_suppress)
        should use this and refrain from re-implementing their own logic.
        """
        return array > threshold

    @staticmethod
    def zero_suppress(array: np.ndarray, threshold: float) -> None:
        """Utility function to zero-suppress a generic array.

        This is returning an array of the same shape of the input where all the
        values lower or equal than the zero suppression threshold are set to zero.

        Arguments
        ---------
        array : array_like
            The input array.

        threshold : float
            The zero suppression threshold.
        """
        mask = np.logical_not(HexagonalReadoutBase.discriminate(array, threshold))
        array[mask] = 0

    @staticmethod
    def latch_timestamp(timestamp: float) -> Tuple[int, int, int]:
        """Latch the event timestamp and return the corresponding fields of the
        digi event contribution: seconds, microseconds and livetime.

        .. warning::
           The livetime calculation is not implemented, yet.

        Arguments
        ---------
        timestamp : float
            The ground-truth event timestamp from the event generator.
        """
        microseconds, seconds = np.modf(timestamp)
        livetime = 0
        return int(seconds), int(1000000 * microseconds), livetime

    def digitize(self, pha: np.ndarray, zero_sup_threshold: int = 0,
        offset: int = 0) -> np.ndarray:
        """Digitize the actual signal.

        Arguments
        ---------
        pha : array_like
            The input array of pixel signals to be digitized.

        zero_sup_threshold : int
            Zero-suppression threshold in ADC counts.

        offset : int
            Optional offset in ADC counts to be applied before the zero suppression.
        """
        # Note that the array type of the input pha argument is not guaranteed, here.
        # Over the course of the calculation the pha is bound to be a float (the noise
        # and the gain are floating-point numbere) before it is rounded to the neirest
        # integer. In order to take advantage of the automatic type casting that
        # numpy implements in multiplication and addition, we use the pha = pha +/*
        # over the pha +/*= form.
        # See https://stackoverflow.com/questions/38673531
        #
        # Add the noise.
        if self.enc > 0:
            pha = pha + rng.generator.normal(0., self.enc, size=pha.shape)
        # ... apply the conversion between electrons and ADC counts...
        pha = pha * self.gain
        # ... round to the neirest integer...
        pha = np.round(pha).astype(int)
        # ... if necessary, add the offset for diagnostic events...
        pha += offset
        # ... zero suppress the thing...
        self.zero_suppress(pha, zero_sup_threshold)
        # ... flatten the array to simulate the serial readout and return the
        # array as the BEE would have.
        return pha.flatten()



class HexagonalReadoutSparse(HexagonalReadoutBase):

    """Description of a pixel sparse readout chip on a hexagonal matrix.
    In the following readout, no ROI is formed, every (and only) triggered pixel of
    the event is kept with its positional information in (col, row) format on the
    hexagonal grid.

    Arguments
    ---------
    layout : HexagonalLayout
        The layout of the hexagonal matrix.

    num_cols : int
        The number of columns in the readout.

    num_rows : int
        The number of rows in the readout.

    pitch : float
        The readout pitch in cm.

    enc : float
        The equivalent noise charge in electrons.

    gain : float
        The readout gain in ADC counts per electron.
    """

    def read(self, timestamp: float, x: np.ndarray, y: np.ndarray, trg_threshold: float,
        zero_sup_threshold: int = 0, offset: int = 0) -> DigiEventSparse:
        """Sparse readout an event.

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

        zero_sup_threshold : int
            Zero suppression threshold in ADC counts.

        offset : int
            Optional offset in ADC counts to be applied before the zero suppression.
        """
        # Sample the input positions over the readout...
        signal = Counter((col, row) for col, row in zip(*self.world_to_pixel(x, y)))
        columns, rows, pha = np.array([[*key, value] for key, value in signal.items()]).T
        # ...apply the trigger...
        trigger_mask = self.discriminate(pha, trg_threshold)
        columns, rows, pha = columns[trigger_mask], rows[trigger_mask], pha[trigger_mask]
        # .. and digitize the pha values.
        pha = self.digitize(pha, zero_sup_threshold, offset)
        seconds, microseconds, livetime = self.latch_timestamp(timestamp)
        self.trigger_id += 1
        return DigiEventSparse(self.trigger_id, seconds, microseconds, livetime, pha, columns, rows)



class HexagonalReadoutRectangular(HexagonalReadoutBase):

    """Description of a pixel readout chip on a hexagonal matrix.
    """

    @staticmethod
    def sum_miniclusters(array: np.ndarray) -> np.ndarray:
        """Sum the values in a given numpy array over its 2 x 2 trigger miniclusters.

        Note that the shape of the target 2-dimensional array must be even in
        both dimensions for the thing to work.
        """
        num_rows, num_cols = array.shape
        return array.reshape((num_rows // 2, 2, num_cols // 2, 2)).sum(-1).sum(1)

    @staticmethod
    def is_odd(value: int) -> bool:
        """Return whether the input integer is odd.

        See https://stackoverflow.com/questions/14651025/ for some metrics about
        the speed of this particular implementation.
        """
        return value & 0x1

    @staticmethod
    def is_even(value: int) -> bool:
        """Return whether the input integer is even.
        """
        return not HexagonalReadoutRectangular.is_odd(value)

    def sample(self, x: np.ndarray, y: np.ndarray) -> Tuple[Tuple[int, int], np.ndarray]:
        """Spatially sample a pair of arrays of x and y coordinates in physical
        space onto logical (hexagonal) coordinates in logical space.

        This is achieved by converting the (x, y) physical coordinates into the
        corresponding (col, row) logical coordinates on the hexagonal grid, and
        then filling a two-dimensional histogram in logical space.

        .. note::

           The output two-dimensional histogram is restricted to the pixels with
           a physical signal, in order to avoid having to deal with large sparse
           arrays downstream. See https://github.com/lucabaldini/hexsample/issues/12
           for more details about the reasoning behind this.

        Arguments
        ---------
        x : array_like
            The physical x coordinates to sample.

        y : array_like
            The physical y coordinates to sample.

        Returns
        -------
        min_col, min_row, signal : 3-element tuple (2 integers and an array)
            The coordinates of the bottom-left corner of the smallest rectangle
            containing all the signal, and the corresponding histogram of the
            signal itself, in electron equivalent.
        """
        # pylint: disable=invalid-name
        col, row = self.world_to_pixel(x, y)
        # Determine the corners of the relevant rectangle where the signal histogram
        # should be built. Reminder: in our trigger minicluster arrangement the minimum
        # column and row coordinates are always even and the maximum column and
        # row coordinates are always odd.
        min_col, max_col, min_row, max_row = col.min(), col.max(), row.min(), row.max()
        if self.is_odd(min_col):
            min_col -= 1
        if self.is_even(max_col):
            max_col += 1
        if self.is_odd(min_row):
            min_row -= 1
        if self.is_even(max_row):
            max_row += 1
        # Streamlined version of a two-dimensional histogram. As obscure as it
        # might seem, this four-liner is significantly faster than a call to
        # np.histogram2d and allows for a substantial speedup in the event generation.
        num_cols = max_col - min_col + 1
        num_rows = max_row - min_row + 1
        index = num_cols * (row - min_row) + (col - min_col)
        signal = np.bincount(index, minlength=num_cols * num_rows).reshape((num_rows, num_cols))
        return min_col, min_row, signal

    def trigger(self, signal: np.ndarray, trg_threshold, min_col: int, min_row: int,
        padding: Padding) -> Tuple[RegionOfInterest, np.ndarray]:
        """Apply the trigger, calculate the region of interest, and pad the
        signal array to the proper dimension.

        .. warning::
           This is still incorrect at the edges of the readout chip, as we are
           not trimming the ROI (and the corresponding arrays) to the physical
           dimensions of the chip.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        # Sum the sampled signal into the 2 x 2 trigger miniclusters.
        trg_signal = self.sum_miniclusters(signal)
        # Zero-suppress the trigger signal below the trigger threshold.
        self.zero_suppress(trg_signal, trg_threshold)
        # This is tricky, and needs to be documented properly---basically we
        # build arrays with all the (minicluster) columns and rows for which
        # at least one minicluster is above threshold. The multiplicative factor
        # of 2 serves the purpose of converting minicluster to pixel coordinates.
        trg_cols = 2 * np.nonzero(trg_signal.sum(axis=0))[0]
        trg_rows = 2 * np.nonzero(trg_signal.sum(axis=1))[0]
        # Build the actual ROI in chip coordinates and initialize the RegionOfInterest
        # object.
        roi_min_col = min_col + trg_cols.min() - padding.left
        roi_max_col = min_col + trg_cols.max() + 1 + padding.right
        roi_min_row = min_row + trg_rows.min() - padding.top
        roi_max_row = min_row + trg_rows.max() + 1 + padding.bottom
        roi = RegionOfInterest(roi_min_col, roi_max_col, roi_min_row, roi_max_row, padding)
        # And now the actual PHA array: we start with all zeroes...
        pha = np.full(roi.shape(), 0.)
        # ...and then we patch the original signal array into the proper submask.
        num_rows, num_cols = signal.shape
        start_row = padding.bottom - trg_rows.min()
        start_col = padding.left - trg_cols.min()
        pha[start_row:start_row + num_rows, start_col:start_col + num_cols] = signal
        # And do not forget to increment the trigger identifier!
        self.trigger_id += 1
        return roi, pha

    def read(self, timestamp: float, x: np.ndarray, y: np.ndarray, trg_threshold: float,
        padding: Padding, zero_sup_threshold: int = 0, offset: int = 0) -> DigiEventRectangular:
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
        min_col, min_row, signal = self.sample(x, y)
        roi, pha = self.trigger(signal, trg_threshold, min_col, min_row, padding)
        pha = self.digitize(pha, zero_sup_threshold, offset)
        seconds, microseconds, livetime = self.latch_timestamp(timestamp)
        return DigiEventRectangular(self.trigger_id, seconds, microseconds, livetime, pha, roi)



class Xpol3(HexagonalReadoutRectangular):

    """Derived class representing the XPOL-III readout chip.
    """

    def __init__(self, enc: float = 20., gain: float = 1.) -> None:
        """Constructor.
        """
        super().__init__(xpol.XPOL1_LAYOUT, *xpol.XPOL3_SIZE, xpol.XPOL_PITCH, enc, gain)



class HexagonalReadoutCircular(HexagonalReadoutBase):

    """Description of a pixel circular readout chip on a hexagonal matrix.
    In the following readout, the maximum PHA pixel is found and the ROI
    formed by that pixel and its 6 adjacent neighbours.
    The standard shape of columns, rows and pha array is then 7, except
    for events on border, that will have len<7.

    Arguments
    ---------
    layout : HexagonalLayout
        The layout of the hexagonal matrix.

    num_cols : int
        The number of columns in the readout.

    num_rows : int
        The number of rows in the readout.

    pitch : float
        The readout pitch in cm.

    enc : float
        The equivalent noise charge in electrons.

    gain : float
        The readout gain in ADC counts per electron.
    """

    NUM_PIXELS = 7

    def read(self, timestamp: float, x: np.ndarray, y: np.ndarray, trg_threshold: float,
        zero_sup_threshold: int = 0, offset: int = 0) -> DigiEventCircular:
        """Circular readout an event.

        Arguments
        ---------
        timestamp : float
            The event timestamp.

        x : float
            The physical x coordinate of the highest pha pixel.

        y : float
            The physical y coordinate of the highest pha pixel.

        trg_threshold : float
            Trigger threshold in electron equivalent.

        zero_sup_threshold : int
            Zero suppression threshold in ADC counts.

        offset : int
            Optional offset in ADC counts to be applied before the zero suppression.
        """

        # Sample the input positions over the readout...
        sparse_signal = Counter((col, row) for col, row in zip(*self.world_to_pixel(x, y)))
        # ...sampling the input position of the highest PHA pixel over the readout...
        # See: https://stackoverflow.com/questions/70094914/max-on-collections-counter
        coord_max = max(sparse_signal, key=sparse_signal.get)
        col_max, row_max = coord_max
        #... and converting it in ADC channel coordinates (value from 0 to 6)...
        adc_max = self.adc_channel(*coord_max)
        # ... creating a 7-elements array containing the PHA of the ADC channels from 0 to 6
        # in increasing order and filling it with PHAs of the highest px and its neigbors...
        pha = np.empty(self.NUM_PIXELS)
        pha[adc_max] = sparse_signal[coord_max]
        # ... identifying the 6 neighbours of the central pixel and saving the signal pixels
        # prepending the cooridnates of the highest one...
        for coords in self.neighbors(*coord_max):
            pha[self.adc_channel(*coords)] = sparse_signal[coords]
        # ...apply the trigger...
        # Not sure the trigger is needed, the highest px passed
        # necessarily the trigger or there is no event
        #trigger_mask = self.discriminate(pha, trg_threshold)
        # .. and digitize the pha values.
        pha = self.digitize(pha, zero_sup_threshold, offset)
        seconds, microseconds, livetime = self.latch_timestamp(timestamp)
        # Do not forget to update the trigger_id!
        self.trigger_id += 1
        #The pha array is always in the order [pha(adc0), pha(adc1), pha(adc2), pha(adc3), pha(adc4), pha(adc5), pha(adc6)]
        return DigiEventCircular(self.trigger_id, seconds, microseconds, livetime, pha, *coords)
