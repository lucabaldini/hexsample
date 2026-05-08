# Copyright (C) 2026 the hexsample team.
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

"""Test suite for hexsample.stats
"""

import numpy as np

import pytest

from hexsample import rng
from hexsample.stats import RunningStats

rng.initialize(seed=666)


def test_running_stats_scalar():
    """Test the scalar version of the running stats.
    """
    running_stats = RunningStats()
    data = rng.generator.normal(size=1000)
    for val in data:
        running_stats.update(val)
    assert np.isclose(running_stats.mean(), np.mean(data))
    assert np.isclose(running_stats.var(), np.var(data, ddof=1))
    assert np.isclose(running_stats.std(), np.std(data, ddof=1))

    # Offset makes no sense for scalar stats, so it should raise an error.
    with pytest.raises(ValueError):
        running_stats.update(0., offset=1)

    # And, for what it's worth, we still support masks for scalar stats.
    running_stats.update(0., mask=True)
    assert running_stats.size() == 1001
    running_stats.update(0., mask=False)
    assert running_stats.size() == 1001

    # Make sure that a mask with the wrong shape raises an error.
    with pytest.raises(ValueError):
        running_stats.update(0., mask=np.array([True, False]))


def test_zero_division():
    """Test that we don't get NaNs when we have zero counts.
    """
    running_stats = RunningStats()
    print(running_stats.std())


def test_running_stats_1d(shape=10):
    """Test the 1D version of the running stats.
    """
    # Generate some random data.
    data = rng.generator.normal(size=(1000, shape))

    # Tier 1: update all the indices in the underlying array.
    running_stats = RunningStats(shape)
    for val in data:
        running_stats.update(val)
    assert np.allclose(running_stats.mean(), np.mean(data, axis=0))
    assert np.allclose(running_stats.var(), np.var(data, axis=0, ddof=1))
    assert np.allclose(running_stats.std(), np.std(data, axis=0, ddof=1))

    # Tier 2: use a smaller array without offsets.
    running_stats = RunningStats(shape)
    lim = shape // 2
    for val in data:
         running_stats.update(val[:lim])
    assert np.allclose(running_stats.mean()[:lim], np.mean(data[:, :lim], axis=0))
    assert np.allclose(running_stats.mean()[lim:], 0.)
    assert np.allclose(running_stats.var()[:lim], np.var(data[:, :lim], axis=0, ddof=1))
    assert np.isnan(running_stats.var()[lim:]).all()

    # Tier 3: use a smaller array with offsets.
    running_stats = RunningStats(shape)
    lim = shape // 2
    offset = shape - lim
    for val in data:
        running_stats.update(val[offset:offset+lim], offset=offset)
    assert np.allclose(running_stats.mean()[lim:], np.mean(data[:, lim:], axis=0))
    assert np.allclose(running_stats.mean()[:lim], 0.)
    assert np.allclose(running_stats.var()[lim:], np.var(data[:, lim:], axis=0, ddof=1))
    assert np.isnan(running_stats.var()[:lim]).all()

    # Tier 4: full_array with mask.
    running_stats = RunningStats(shape)
    for val in data:
        running_stats.update(val, mask=val > 0)


def test_running_stats_2d(shape=(3, 3)):
    """Test the 2D version of the running stats.
    """
    # Generate some random data.
    data = rng.generator.normal(size=(1000, *shape))

    # Tier 1: update all the indices in the underlying array.
    running_stats = RunningStats(shape)
    for val in data:
        running_stats.update(val)
    assert np.allclose(running_stats.mean(), np.mean(data, axis=0))
    assert np.allclose(running_stats.var(), np.var(data, axis=0, ddof=1))
    assert np.allclose(running_stats.std(), np.std(data, axis=0, ddof=1))