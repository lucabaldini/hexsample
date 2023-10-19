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

"""Test suite for hexsample.rng
"""

import numpy as np
import pytest

from hexsample import rng


def test_generator():
    """Basic test of the global generator.
    """
    # Make sure we start from the uninitialize state.
    rng.reset()
    # Before calling the initialize() function we shall get a RuntimeError
    # at any attempt of doing anything with the generator.
    with pytest.raises(RuntimeError):
        a = rng.generator.normal()
    # Initialization to non-default values.
    bit_generator_class = np.random.PCG64
    seed = 28
    rng.initialize(bit_generator_class, seed)
    assert isinstance(rng.generator._bit_generator, bit_generator_class)
    assert rng.generator._bit_generator._seed_seq.entropy == seed
    # Default initizalization.
    rng.initialize()
    assert isinstance(rng.generator._bit_generator, rng.DEFAULT_BIT_GENERATOR)
