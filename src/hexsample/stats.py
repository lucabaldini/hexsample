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
        """

    def _check_rank(self, val: np.ndarray) -> None:
        """
        """
        val = np.asarray(val)
        if val.ndim != self._counts.ndim:
            raise ValueError(f"Expected {self._counts.ndim}D array, got {val.ndim}D array.")

    def _update_scalar(self, val: float) -> None:
        """
        """
        self._counts += 1
        delta = val - self._mean
        self._mean += delta / self._counts
        self._m2 += delta * (val - self._mean)

    def _update_array(self, val: np.ndarray, region: Union[slice, Tuple[slice, ...]],
                      mask: np.ndarray = None) -> None:
        """
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


class _RunningStats1d(AbstractRunningStats):

    def update(self, val: np.ndarray, offset: int = 0, mask: np.ndarray = None) -> None:
        """Overloaded abstract method.
        """
        self._check_rank(val)
        region = slice(offset, offset + len(val))
        self._update_array(val, region, mask)


class _RunningStatsArray(AbstractRunningStats):

    def update(self, val: np.ndarray, offset: Tuple[int, ...] = None,
               mask: np.ndarray = None) -> None:
        """
        """
        self._check_rank(val)
        if offset is None:
            offset = offset or tuple(0 for _ in val.shape)
        region = tuple(slice(pos, pos + dim) for pos, dim in zip(offset, val.shape))
        self._update_array(val, region, mask)


def RunningStats(shape: Union[int, Tuple[int, ...]] = ()) -> AbstractRunningStats:
    """Factory function for the scalar version of the running stats.
    """
    if shape == ():
        return _RunningStatsScalar(shape)
    elif isinstance(shape, int) or len(shape) == 1:
        return _RunningStats1d(shape)
    else:
        return _RunningStatsArray(shape)