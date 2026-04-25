# Copyright (C) 2023--2025 the hexsample team.
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

"""Readout facilities.
"""

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import numpy as np

from . import rng
from .base import TypeProxy
from .digi import DigiEventBase, DigiEventCircular, DigiEventRectangular
from .hexagon import HexagonalGrid
from .roi import Padding, RegionOfInterest

if TYPE_CHECKING:
    from hexsample.calibration import CalibrationMatrix



class AbstractReadout(ABC):

    """Abstract base class for a generic pixel readout chip.

    This is a simple abstract class defining the a single static method read(), along
    with the associated signature, that all readout chips must implement.
    """

    @abstractmethod
    def read(self, timestamp: float, x: np.ndarray, y: np.ndarray) -> DigiEventBase:
        """Readout a single event, given the input coordinates of the charge.

        Arguments
        ---------
        timestamp : float
            The event timestamp.

        x : array_like
            The physical x coordinates of the input charge.

        y : array_like
            The physical y coordinates of the input charge.
        """


@dataclass
class HexagonalReadoutBase(HexagonalGrid, AbstractReadout):

    """Description of a pixel readout chip on a hexagonal matrix.

    Note that, in addition to the physical properties (noise and gain) the readout
    chip comes up at creation time with a well defined configuration, in terms of
    trigger and zero-suppression thresholds. Arguably, in the future we might add the
    possibility to change these parameters at runtime to make things more germane to
    what happens in real life, but for the time being that seems hardly necessary.

    Also note the readout chip inherits all the members from HexagonalGrid.

    Arguments
    ---------
    enc : CalibrationMatrix
        The equivalent noise charge in electrons.

    gain : CalibrationMatrix
        The readout gain in ADC counts per electron (default 1, which means that
        the PHA you get out are the electrons collected).

    pedestal : CalibrationMatrix
        The pedestal in ADC counts.

    trg_threshold : float
        Trigger threshold in electron equivalent (note this is a float because it
        is expressed in physical space, not in electronics space).

    zero_sup_threshold : int
        Zero suppression threshold in ADC counts.
    """

    enc: Optional["CalibrationMatrix"] = None
    gain: Optional["CalibrationMatrix"] = None
    pedestal: Optional["CalibrationMatrix"] = None
    trg_threshold: float = 500.
    zero_sup_threshold: int = 0

    def __post_init__(self):
        """Post-initialization.
        """
        HexagonalGrid.__post_init__(self)
        if self.enc is None or self.gain is None:
            raise TypeError("enc and gain are mandatory and cannot be None")
        self.trigger_id = -1

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
        return not value & 0x1

    @staticmethod
    def discriminate(array: np.ndarray, threshold: float) -> np.ndarray:
        """Utility function acting as a simple constant-threshold discriminator
        over a generic array. This returns a boolean mask with True for all the
        array elements larger than the threshold.

        This is intented to avoid possible confusion between strict and loose
        comparison operators (e.g., < vs <=) when comparing the content of an array
        with a threshold, and all functions downstream doing this (e.g., zero_suppress)
        should use this and refrain from re-implementing their own logic.

        Note this is a static method that can be used interchangeably, e.g., to operate
        on the pulse height array or the trigger signal array.
        """
        return array > threshold

    @staticmethod
    def zero_suppress(array: np.ndarray, threshold: float) -> None:
        """Utility function to zero-suppress a generic array.

        This is returning an array of the same shape of the input where all the
        values lower or equal than the zero suppression threshold are set to zero.

        Note this is a static method that can be used interchangeably, e.g., to operate
        on the pulse height array or the trigger signal array.

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

    def digitize(self, pha: np.ndarray,
                 coords: Union[RegionOfInterest, Sequence[Tuple[int, int]]]) -> np.ndarray:
        """Digitize the actual signal.

        Arguments
        ---------
        pha : array_like
            The input array of pixel signals to be digitized.
        """
        # Create cols and rows arrays from the input coordinates, depending on the readout mode.
        # In case of rectangular readout, we have a RegionOfInterest, otherwise we have a list with
        # the coordinates of the pixels to be read out.
        if isinstance(coords, RegionOfInterest):
            cols, rows = coords.readout_slice()
        else:
            cols, rows = np.array(coords).T
        # Add the noise
        noise = rng.generator.normal(0., scale=self.enc(cols, rows))
        pha = pha + noise
        # Apply the conversion between electrons and ADC counts
        pha = pha * self.gain(cols, rows)
        # Round to the nearest integer
        pha = np.round(pha).astype(int)
        # Add the pedestal
        pha += self.pedestal(cols, rows)
        # Zero suppress the thing.
        self.zero_suppress(pha, self.zero_sup_threshold)
        # Flatten the array to simulate the serial readout and return the
        # array as the BEE would have.
        return pha.flatten()


class HexagonalReadoutCircular(HexagonalReadoutBase):

    """Fixed 7-pixel ROI readout chip on a hexagonal matrix.

    This class mimics a readout chip with a fixed 7-pixel ROI on a hexagonal matrix,
    where the maximum PHA is found and the ROI formed by that pixel and its 6 adjacent
    neighbours.

    Note events on border will have less than 7 pixels.
    """

    NUM_PIXELS = 7

    def read(self, timestamp: float, x: np.ndarray, y: np.ndarray) -> DigiEventCircular:
        """Overloaded method.
        """
        # Sample the input positions over the readout...
        sparse_signal = Counter((col, row) for col, row in zip(*self.world_to_pixel(x, y)))
        # ...sampling the input position of the highest PHA pixel over the readout...
        # See: https://stackoverflow.com/questions/70094914/max-on-collections-counter
        coord_max = max(sparse_signal, key=sparse_signal.get)
        # col_max, row_max = coord_max
        #... and converting it in ADC channel coordinates (value from 0 to 6)...
        adc_max = self.adc_channel(*coord_max)
        # ... creating a 7-elements array containing the PHA of the ADC channels from 0 to 6
        # in increasing order and filling it with PHAs of the highest px and its neigbors...
        pha = np.empty(self.NUM_PIXELS)
        pha[adc_max] = sparse_signal[coord_max]
        # ... identifying the 6 neighbors of the central pixel and saving the signal pixels
        # prepending the coordinates of the highest one...
        coords = [coord_max]
        for _coords in self.neighbors(*coord_max):
            pha[self.adc_channel(*_coords)] = sparse_signal[_coords]
            coords.append(_coords)
        # Not sure the trigger is needed, the highest px passed
        # necessarily the trigger or there is no event
        # trigger_mask = self.discriminate(pha, self.trg_threshold)
        # .. and digitize the pha values.
        pha = self.digitize(pha, coords)
        seconds, microseconds, livetime = self.latch_timestamp(timestamp)
        # And do not forget to increment the trigger identifier!
        self.trigger_id += 1
        # The pha array is always in the order
        # [pha(adc0), pha(adc1), pha(adc2), pha(adc3), pha(adc4), pha(adc5), pha(adc6)]
        return DigiEventCircular(self.trigger_id, seconds, microseconds, livetime, pha, *coord_max)


@dataclass
class HexagonalReadoutRectangular(HexagonalReadoutBase):

    """ROI-based readout chip on a hexagonal matrix.

    This mimics the functionality of the various generations of XPOL readout chips,
    where a ROT (region of trigger) is automatically defines based on the 2 x 2
    minicluster trigger logic, and then expanded with a configurable padding to form
    the ROI (region of interest) that is actually read out.
    """

    padding: Padding = Padding(2, 2, 2, 2)

    @staticmethod
    def sum_miniclusters(array: np.ndarray) -> np.ndarray:
        """Sum the values in a given numpy array over its 2 x 2 trigger miniclusters.

        Note that the shape of the target 2-dimensional array must be even in
        both dimensions for the thing to work.
        """
        num_rows, num_cols = array.shape
        return array.reshape((num_rows // 2, 2, num_cols // 2, 2)).sum(-1).sum(1)

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

    def trigger(self, signal: np.ndarray, min_col: int,
                min_row: int) -> Tuple[RegionOfInterest, np.ndarray]:
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
        self.zero_suppress(trg_signal, self.trg_threshold)
        # This is tricky, and needs to be documented properly---basically we
        # build arrays with all the (minicluster) columns and rows for which
        # at least one minicluster is above threshold. The multiplicative factor
        # of 2 serves the purpose of converting minicluster to pixel coordinates.
        trg_cols = 2 * np.nonzero(trg_signal.sum(axis=0))[0]
        trg_rows = 2 * np.nonzero(trg_signal.sum(axis=1))[0]
        # Build the actual ROI in chip coordinates and initialize the RegionOfInterest
        # object.
        roi_min_col = min_col + trg_cols.min() - self.padding.left
        roi_max_col = min_col + trg_cols.max() + 1 + self.padding.right
        roi_min_row = min_row + trg_rows.min() - self.padding.top
        roi_max_row = min_row + trg_rows.max() + 1 + self.padding.bottom
        roi = RegionOfInterest(roi_min_col, roi_max_col, roi_min_row, roi_max_row, self.padding)
        # And now the actual PHA array: we start with all zeroes...
        pha = np.full(roi.shape(), 0.)
        # ...and then we patch the original signal array into the proper submask.
        num_rows, num_cols = signal.shape
        start_row = self.padding.top - trg_rows.min()
        start_col = self.padding.left - trg_cols.min()
        pha[start_row:start_row + num_rows, start_col:start_col + num_cols] = signal
        # And do not forget to increment the trigger identifier!
        self.trigger_id += 1
        return roi, pha

    def read(self, timestamp: float, x: np.ndarray, y: np.ndarray) -> DigiEventRectangular:
        """Overloaded method.
        """
        # pylint: disable=invalid-name, too-many-arguments
        min_col, min_row, signal = self.sample(x, y)
        roi, pha = self.trigger(signal, min_col, min_row)
        pha = self.digitize(pha, roi)
        seconds, microseconds, livetime = self.latch_timestamp(timestamp)
        return DigiEventRectangular(self.trigger_id, seconds, microseconds, livetime, pha, roi)


# Definition of the readout proxy.
ReadoutProxy = TypeProxy("readout_mode") # pylint: disable=invalid-name
ReadoutProxy.register("circular", HexagonalReadoutCircular)
ReadoutProxy.register("rectangular", HexagonalReadoutRectangular)


# TODO: TO BE REMOVED!

class HexagonalReadoutMode(str, Enum):
    """Enum class expressing the possible readout strategies.
    """

    RECTANGULAR = "rectangular"
    CIRCULAR = "circular"


# Mapping for the readout chip classes for each readout mode.
_READOUT_CLASS_DICT = {
    HexagonalReadoutMode.RECTANGULAR: HexagonalReadoutRectangular,
    HexagonalReadoutMode.CIRCULAR: HexagonalReadoutCircular
}

def _readout_class(mode: HexagonalReadoutMode) -> type:
    """Return the proper class to be used to instantiate a readout chip for a given
    readout mode.
    """
    return _READOUT_CLASS_DICT[mode]

def readout_chip(mode: HexagonalReadoutMode, *args, **kwargs):
    """Return an instance of the proper readout chip for a given readout mode.
    """
    return _readout_class(mode)(*args, **kwargs)
