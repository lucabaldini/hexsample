# Copyright (C) 2026 the hexsample team.
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

"""Statistical tools.
"""

from typing import Tuple, Union

import numpy as np


class RunningStats:

    """Small convenience class to accumulate running statistics (mean and variance)
    of a stream of data.

    This is designed to work both for scalar values and for arrays of arbitrary
    shape, and uses the Welford algorithm to perform the computation.

    Arguments
    ---------
    shape : int or tuple of ints, optional
        The shape of the underlying arrays for accumulating the statistics.
    """

    def __init__(self, shape: Union[int, Tuple[int, ...]] = ()) -> None:
        """Constructor.
        """
        # Normalize the input shape to a tuple and initialize the internal arrays.
        if isinstance(shape, int):
            shape = (shape,)
        self._rank = len(shape)
        self._counts = np.zeros(shape, dtype=np.int64)
        self._mean = np.zeros(shape, dtype=np.float64)
        self._m2 = np.zeros(shape, dtype=np.float64)

    def update(self, val: np.ndarray, offset: Union[int, Tuple[int, ...]] = None,
               mask: np.ndarray = None) -> None:
        """Update the running statistics.
        """
        # Check the rank of the values for the update.
        val = np.asarray(val)
        if val.ndim != self._rank:
            raise ValueError(f"Expected {self._rank}D array, got {val.ndim}D array.")

        # Calculate the sub-region for the update based on the offset.
        if offset is None:
            region = ... if self._rank == 0 else tuple(slice(0, dim) for dim in val.shape)
        else:
            if self._rank == 0:
                raise ValueError("Offset is not meaningful for scalar running stats.")
            if isinstance(offset, int):
                offset = (offset,)
            if len(offset) != self._rank:
                raise ValueError(f"Expected offset with rank {self._rank}.")
            region = tuple(slice(pos, pos + dim) for pos, dim in zip(offset, val.shape))

        # Do the actual calculation.
        counts = self._counts[region]
        mean = self._mean[region]
        m2 = self._m2[region]
        if mask is None:
            counts += 1
            delta = val - mean
            mean += delta / counts
            m2 += delta * (val - mean)
        else:
            counts[mask] += 1
            delta = val[mask] - mean[mask]
            mean[mask] += delta / counts[mask]
            m2[mask] += delta * (val[mask] - mean[mask])

    def counts(self) -> np.ndarray:
        """Return the current value for the counts.
        """
        return self._counts

    def mean(self) -> np.ndarray:
        """Return the current value for the mean.
        """
        return self._mean

    def var(self, ddof: int = 1):
        """Return the current value for the variance.

        Arguments
        ---------
        ddof : int
            Delta degrees of freedom (default is 1 for the unbiased sample variance.)
        """
        return self._m2 / (self._counts - ddof)

    def std(self, ddof: int = 1):
        """Return the current value for the standard deviation.

        Arguments
        ---------
        ddof : int
            Delta degrees of freedom (default is 1.)
        """
        return np.sqrt(self.var(ddof))
