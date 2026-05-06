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
    """

    def __init__(self, shape: Union[int, Tuple[int, ...]] = ()) -> None:
        """Constructor.
        """
        self._mean = np.zeros(shape, dtype=np.float64)
        self._counts = np.zeros(shape, dtype=np.int64)
        self._m2 = np.zeros(shape, dtype=np.float64)

    def update(self, val: np.ndarray, *indices: np.ndarray) -> None:
        """Update the running stats.

        Arguments
        ---------
        val : array-like
            The new value(s) to be included in the running stats.

        *indices : array-like
            The indices of the elements to update. (Note the shape of the arguments
            must match that of the values for the update.)
        """
        val = val[*indices]
        self._counts[*indices] += 1
        delta = val - self._mean[*indices]
        self._mean[*indices] += delta / self._counts[*indices]
        self._m2[*indices] += delta * (val - self._mean[*indices])

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