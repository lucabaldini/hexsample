:mod:`~hexsample.base` --- Utilities
====================================

This module provides generic helpers and utilities for the hexsample package.

:class:`~hexsample.base.AbstractRandomGenerator` is an abstract base class for
any class that implements random number generation functionality. It defines
a single abstract method, :meth:`~hexsample.base.AbstractRandomGenerator.rvs`
that must be implemented by any subclass. All the spectrum and beam objects in
the :mod:`~hexsample.source` module use this interface.

:class:`~hexsample.base.AbstractPlottable` is an abstract base class for any class
that implements plotting functionality. It defines a single abstract method
:meth:`~hexsample.base.AbstractPlottable.render` that must be implemented by any
subclass and is called in the :meth:`~hexsample.base.AbstractPlottable.plot`
method to handle matplotlib axes transparently. All the spectrum objects in
the :mod:`~hexsample.source` module use this interface.

:class:`~hexsample.base.TypeProxy` is a concrete class that acts as a dispatcher
for different types of objects. This is used when we have to instantiate different
(but related) classes based on user input, as in spectra and beams.
This is best illustrated in the the following example, which is taken from the
:mod:`~hexsample.source` module.

.. code-block:: python

    BeamProxy = TypeProxy("beam") # pylint: disable=invalid-name
    BeamProxy.register("point", PointBeam)
    BeamProxy.register("disk", DiskBeam)
    BeamProxy.register("gaussian", GaussianBeam, default=True)
    BeamProxy.register("triangular", TriangularBeam)
    BeamProxy.register("hexagonal", HexagonalBeam)


The ``BeamProxy`` object can be used to instantiate different beam types based on
user input through keyword arguments, using the
:meth:`hexsample.base.TypeProxy.from_filtered_kwargs` method, e.g.

.. code-block:: python

    from hexsample.source import BeamProxy

    # Instantiate a Gaussian beam
    beam = BeamProxy.from_filtered_kwargs(beam="gaussian", sigma=1.0)

.. warning::

    The ``TypeProxy`` mechanism is fragile in that, by definition, is not protected
    against errors (e.g., typos) in the keyword arguments. It is mainly intended
    to be used in conjunction with the command-line interface of the package, in
    which case everything is fine, as all basic sanity checks are performed on the
    cli side, but you should always prefer explicit instantiation of the classes
    when using the package programmatically.

In addition, the ``TypeProxy`` class provides methods to list the registered types
(:meth:`hexsample.base.TypeProxy.choices`) and to get the default type
(:meth:`hexsample.base.TypeProxy.default`), which are handy for the
implementation of the command-line interface.



Module documentation
--------------------

.. automodule:: hexsample.base
