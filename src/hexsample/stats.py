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

from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np


class AbstractRunningStats(ABC):

    """Small convenience class to accumulate running statistics (mean and variance)
    of a stream of data.

    This is a base abstract class designed to work both for scalar values and for
    arrays of arbitrary shape. The actual implementation is delegated to the
    concrete subclasses, which are instantiated by the factory function `RunningStats`
    based on the provided shape.

    Arguments
    ---------
    shape : int or tuple of ints, optional
        The shape of the underlying arrays for accumulating the statistics.
    """

    def __init__(self, shape: Union[int, Tuple[int, ...]] = ()) -> None:
        """Constructor.
        """
        self._counts = np.zeros(shape, dtype=np.int64)
        self._mean = np.zeros(shape, dtype=np.float64)
        self._m2 = np.zeros(shape, dtype=np.float64)

    @abstractmethod
    def update(self, val, *args, **kwargs) -> None:
        """Update the running statistics with a new value.

        This needs to be implemented by the concrete subclasses.
        """

    def _check_rank(self, val: np.ndarray) -> None:
        """Check that the input value(s) has the expected rank.

        This raises a ValueError if the input value does not have the same number
        of dimensions as the underlying arrays.

        Arguments
        ---------
        val : array-like
            The input value to check.
        """
        val = np.asarray(val)
        if val.ndim != self._counts.ndim:
            raise ValueError(f"Expected {self._counts.ndim}D array, got {val.ndim}D array.")

    def _update_scalar(self, val: float) -> None:
        """Update the running statistics with a new scalar value.

        Arguments
        ---------
        val : float
            The new scalar value to incorporate.
        """
        self._counts += 1
        delta = val - self._mean
        self._mean += delta / self._counts
        self._m2 += delta * (val - self._mean)

    def _update_array(self, val: np.ndarray, region: Union[slice, Tuple[slice, ...]],
                      mask: np.ndarray = None) -> None:
        """Update the running statistics with a new array of values.

        Arguments
        ---------
        val : array-like
            The new array of values to incorporate.
        """
        # Cache the necessary views to avoid repeated indexing.
        counts = self._counts[region]
        mean = self._mean[region]
        m2 = self._m2[region]
        # Do the actual calculation.
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

        """
        return np.sqrt(self.var(ddof))


class _RunningStatsScalar(AbstractRunningStats):

    def update(self, val: float) -> None:
        """Overloaded abstract method.
        """
        self._check_rank(val)
        self._update_scalar(val)


class _RunningStatsArray(AbstractRunningStats):

    def update(self, val: np.ndarray, offset: Tuple[int, ...] = None,
               mask: np.ndarray = None) -> None:
        """Overloaded abstract method.
        """
        self._check_rank(val)
        if offset is None:
            offset = offset or tuple(0 for _ in val.shape)
        elif isinstance(offset, int):
            offset = (offset,)
        region = tuple(slice(pos, pos + dim) for pos, dim in zip(offset, val.shape))
        self._update_array(val, region, mask)


def RunningStats(shape: Union[int, Tuple[int, ...]] = ()) -> AbstractRunningStats:
    """Factory function for the scalar version of the running stats.
    """
    if shape == ():
        return _RunningStatsScalar(shape)
    else:
        return _RunningStatsArray(shape)