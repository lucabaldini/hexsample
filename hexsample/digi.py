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

"""Digitization facilities.
"""

from dataclasses import dataclass
from typing import Tuple

from loguru import logger
import numpy as np

from hexsample import rng
from hexsample.hexagon import HexagonalGrid, HexagonalLayout
from hexsample.pprint import AnsiFontEffect, ansi_format, space, line
from hexsample.roi import Padding, RegionOfInterest
from hexsample import xpol



@dataclass
class DigiEvent:

    """Descriptor for a digitized event.

    A digitized event is the datum of a RegionOfInterest object, a trigger
    identifier, a timestamp and a 1-dimensional array of ADC counts, in the
    readout order. Note that the length of the pha array must match the size of
    the ROI.

    Note that, since in the simulated digitization process we typically create
    an ROI first, and only in a second moment a fully fledged event, we prefer
    composition over inheritance, and deemed more convenient to have a
    :class:`RegionOfInterest <hexsample.roi.RegionOfInterest>` object as a class
    member, rather than inherit from :class:`RegionOfInterest <hexsample.roi.RegionOfInterest>`.

    Arguments
    ---------
    trigger_id : int
        The trigger identifier.

    seconds : int
        The integer part of the timestamp.

    microseconds : int
        The fractional part of the timestamp.

    roi : RegionOfInterest
        The region of interest for the event.

    pha : np.ndarray
        The pixel content of the event, in the form of a 1-dimensional array
        whose length matches the size of the ROI.
    """

    trigger_id : int
    seconds : int
    microseconds : int
    livetime : int
    roi : RegionOfInterest
    pha : np.ndarray

    def __post_init__(self) -> None:
        """Post-initialization code.

        Here we reshape the one-dimensional PHA array coming from the serial
        readout to the proper ROI shape for all subsequent operations.
        """
        try:
            self.pha = self.pha.reshape(self.roi.shape())
        except ValueError as error:
            logger.error(f'Error in {self.__class__.__name__} post-initializaion.')
            print(self.roi)
            print(f'ROI size: {self.roi.size}')
            print(f'pha size: {self.pha.size}')
            logger.error(error)

    def __eq__(self, other) -> bool:
        """Overloaded comparison operator.
        """
        return (self.trigger_id, self.seconds, self.microseconds, self.livetime) == \
            (other.trigger_id, other.seconds, other.microseconds, other.livetime) and \
            self.roi == other.roi and np.allclose(self.pha, other.pha)

    @classmethod
    def from_digi(cls, row : np.ndarray, pha : np.ndarray):
        """Alternative constructor rebuilding an object from a row on a digi file.

        This is used internally when we access event data in a digi file, and
        we need to reassemble a DigiEvent object from a given row of a digi table.
        """
        # pylint: disable=too-many-locals
        trigger_id, seconds, microseconds, livetime, min_col, max_col, min_row, max_row,\
            pad_top, pad_right, pad_bottom, pad_left = row
        padding = Padding(pad_top, pad_right, pad_bottom, pad_left)
        roi = RegionOfInterest(min_col, max_col, min_row, max_row, padding)
        return cls(trigger_id, seconds, microseconds, livetime, roi, pha)

    def __call__(self, col : int, row : int) -> int:
        """Retrieve the pha content of the event for a given column and row.

        Internally this is subtracting the proper offset to the column and row
        indexes in order to convert from chip coordinates to indexes of the
        underlying PHA array. Note that, due to the way arrays are indexed in numpy,
        we do need to swap the column and the row.

        Arguments
        ---------
        col : int
            The column number (in chip coordinates).

        row : int
            The row number (in chip coordinates).
        """
        return self.pha[row - self.roi.min_row, col - self.roi.min_col]

    def highest_pixel(self, absolute : bool = True) -> Tuple[int, int]:
        """Return the coordinates (col, row) of the highest pixel.

        Arguments
        ---------
        absolute : bool
            If true, the absolute coordinates (i.e., those referring to the readout
            chip) are returned; otherwise the coordinates are intended relative
            to the readout window (i.e., they can be used to index the pha array).
        """
        # Note col and row are swapped, here, due to how the numpy array are indexed.
        # pylint: disable = unbalanced-tuple-unpacking
        row, col = np.unravel_index(np.argmax(self.pha), self.pha.shape)
        if absolute:
            col += self.roi.min_col
            row += self.roi.min_row
        return col, row

    def timestamp(self) -> float:
        """Return the timestamp of the event, that is, the sum of the second and
        microseconds parts of the DigiEvent contributions as a floating point number.
        """
        return self.seconds + 1.e-6 * self.microseconds

    def ascii(self, pha_width : int = 5):
        """Ascii representation.
        """
        fmt = f'%{pha_width}d'
        cols = self.roi.col_indexes()
        rows = self.roi.row_indexes()
        big_space = space(2 * pha_width + 1)
        text = f'\n{big_space}'
        text += ''.join([fmt % col for col in cols])
        text += f'\n{big_space}'
        text += ''.join([fmt % (col - self.roi.min_col) for col in cols])
        text += f'\n{big_space}+{line(pha_width * self.roi.num_cols)}\n'
        for row in rows:
            text += f'{fmt % row} {fmt % (row - self.roi.min_row)}|'
            for col in cols:
                pha = fmt % self(col, row)
                if not self.roi.in_rot(col, row):
                    pha = ansi_format(pha, AnsiFontEffect.FG_BRIGHT_BLUE)
                text += pha
            text += f'\n{big_space}|\n'
        text += f'{self.roi}\n'
        return text



class HexagonalReadout(HexagonalGrid):

    """Description of a pixel readout chip on a hexagonal matrix.
    """

    def __init__(self, layout : HexagonalLayout, num_cols : int, num_rows : int,
                 pitch : float, enc : float, gain : float) -> None:
        """Constructor.
        """
        # pylint: disable=too-many-arguments
        super().__init__(layout, num_cols, num_rows, pitch)
        self.enc = enc
        self.gain = gain
        self.shape = (self.num_rows, self.num_cols)
        self.trigger_id = -1

    @staticmethod
    def sum_miniclusters(array : np.ndarray) -> np.ndarray:
        """Sum the values in a given numpy array over its 2 x 2 trigger miniclusters.

        Note that the shape of the target 2-dimensional array must be even in
        both dimensions for the thing to work.
        """
        num_rows, num_cols = array.shape
        return array.reshape((num_rows // 2, 2, num_cols // 2, 2)).sum(-1).sum(1)

    @staticmethod
    def zero_suppress(array : np.ndarray, threshold : float) -> None:
        """Utility function to zero-suppress an generic array.

        This is returning an array of the same shape of the input where all the
        values lower or equal than the zero suppression threshold are set to zero.

        Arguments
        ---------
        array : array_like
            The input array.

        threshold : float
            The zero suppression threshold.
        """
        array[array <= threshold] = 0

    @staticmethod
    def is_odd(value : int) -> bool:
        """Return whether the input integer is odd.

        See https://stackoverflow.com/questions/14651025/ for some metrics about
        the speed of this particular implementation.
        """
        return value & 0x1

    @staticmethod
    def is_even(value : int) -> bool:
        """Return whether the input integer is even.
        """
        return not HexagonalReadout.is_odd(value)

    def sample(self, x : np.ndarray, y : np.ndarray) -> Tuple[Tuple[int, int], np.ndarray]:
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

    def trigger(self, signal : np.ndarray, trg_threshold, min_col : int, min_row : int,
        padding : Padding) -> Tuple[RegionOfInterest, np.ndarray]:
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

    def digitize(self, pha : np.ndarray, zero_sup_threshold : int = 0,
        offset : int = 0) -> np.ndarray:
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
        # Add the noise.
        if self.enc > 0:
            pha += rng.generator.normal(0., self.enc, size=pha.shape)
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

    @staticmethod
    def latch_timestamp(timestamp : float) -> Tuple[int, int, int]:
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
        min_col, min_row, signal = self.sample(x, y)
        roi, pha = self.trigger(signal, trg_threshold, min_col, min_row, padding)
        pha = self.digitize(pha, zero_sup_threshold, offset)
        seconds, microseconds, livetime = self.latch_timestamp(timestamp)
        return DigiEvent(self.trigger_id, seconds, microseconds, livetime, roi, pha)



class Xpol3(HexagonalReadout):

    """Derived class representing the XPOL-III readout chip.
    """

    def __init__(self, enc : float = 20., gain : float = 1.) -> None:
        """Constructor.
        """
        super().__init__(xpol.XPOL1_LAYOUT, *xpol.XPOL3_SIZE, xpol.XPOL_PITCH, enc, gain)
