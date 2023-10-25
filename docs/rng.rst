:mod:`hexsample.rng` --- Random numbers
=======================================

This module is an attempt to move away from the legacy numpy
`RandomState <https://numpy.org/doc/stable/reference/random/legacy.html>`_ and
use the best practices described in the
numpy `random documentation <https://numpy.org/doc/stable/reference/random/index.html>`_
and in `NEP 19 <https://numpy.org/neps/nep-0019-rng-policy.html>`_.

Although the bottomline of the numpy documentation is not to have a global state,
never seed an existing generator, and create the desired generator at the beginning
of the program and then just pass the object around, we deemed changing constructors
(or the signature of the relevant class methods) for all the classes that need to
throw random number too intrusive. Instead, we created this module that has a global
`Generator <https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator>`_
object.

The basic usage of the module is as follows

.. code-block:: python

   from hexsample import rng

   rng.initialize()
   z = rng.generator.normal()

and any attempt to do something with the ``rng.generator`` object will raise a
``RuntimeError`` if the module is not properly initialized. The initialization
function :meth:`initialize() <hexsample.rng.initialize>` is the place where the
particular algorithm for the random number generation, as well as the seed, can be set.
This should be normally called once at the beginning of the program, and then the
global ``rng.generator`` object will be available from everywhere in the package.

The default undelrying bit generator object that we use is the
`SFC64 <https://numpy.org/doc/stable/reference/random/bit_generators/sfc64.html>`_
Small Fast Chaotic PRNG, that is known to be very fast and have good statistical
properties.


Module documentation
--------------------

.. automodule:: hexsample.rng
