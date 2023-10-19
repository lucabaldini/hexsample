# Copyright (C) 2023 luca.baldini@pi.infn.it
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""Application-wide pseudo-random number generator.
"""

import numpy as np

from hexsample import logger


# Default bit generator class---this is the workhorse object that draw random
# numbers equidistributed in the [0, 1] interval.
# pylint: disable=invalid-name
DEFAULT_BIT_GENERATOR = np.random.SFC64



class UninitializedGenerator:

    """Mock class raising an exception at any attempt at an interaction.

    This is a poor-man trick not to let the user forget that the they have to
    call the initialize() function below before being able to draw any random
    number.
    """

    # pylint: disable=too-few-public-methods

    def __getattr__(self, name):
        """Basic hook to implement a no-op class.
        """
        raise RuntimeError('Random number generator not initialized.')



def reset() -> None:
    """Set the generator global object to the uninitialized state.
    """
    global generator
    generator = UninitializedGenerator()

def initialize(bit_generator_class : type = DEFAULT_BIT_GENERATOR, seed : int = None) -> None:
    """Create a random generator from a given underlying bit generator and a given seed.

    This is using the recommended constructor for the random number class Generator,
    and goes toward the philosophy that it is better to create a new generator
    rather than seed one multiple times.

    The available bit generators are:

    * MT19937
    * PCG64
    * PCG64DXSM
    * Philox
    * SFC64

    along with the old RandomState that is kept for compatibility reasons.

    The merits and demerits and the performance of the various methods are briefly
    discussed at:
    https://numpy.org/doc/stable/reference/random/performance.html

    The recommended generator for general use is PCG64 or its upgraded variant PCG64DXSM
    for heavily-parallel use cases. They are statistically high quality, full-featured,
    and fast on most platforms, but somewhat slow when compiled for 32-bit processes.

    Philox is fairly slow, but its statistical properties have very high quality, and it
    is easy to get an assuredly-independent stream by using unique keys.

    SFC64 is statistically high quality and very fast. However, it lacks jumpability.
    If you are not using that capability and want lots of speed, even on 32-bit processes,
    this is your choice.

    MT19937 fails some statistical tests and is not especially fast compared to modern PRNGs.
    For these reasons, we mostly do not recommend using it on its own, only through
    the legacy RandomState for reproducing old results. That said, it has a very long
    history as a default in many systems.

    Arguments
    ---------
    bit_generator_class : type
        The class for the underlying bit generator.

    seed : int
        The initial seed (default to None).
    """
    # pylint: disable=global-statement
    global generator
    seed_sequence = np.random.SeedSequence(seed)
    logger.info(f'Random seed set to {seed_sequence.entropy}')
    bit_generator = bit_generator_class(seed_sequence)
    logger.info(f'Creating new {bit_generator.__class__.__name__} pseudo-random generator...')
    generator = np.random.default_rng(bit_generator)



reset()
