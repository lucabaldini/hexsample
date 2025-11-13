:mod:`hexsample.source` --- X-ray sources
=========================================

This module contains all the facilities for the description of simple X-ray sources,
for both the morphological and spectral part. At the very basic level, a
:class:`Source <hexsample.source.Source>` object is the datum of an energy spectrum
and a beam description, and has the primary purpose of generating lists of X-ray
basic properties (time, energy and position in the horizontal plane) than can be
folded into an actual detector simulation.

.. code-block:: python

  spectrum = LineForest('Cu', 'K')
  beam = GaussianBeam(sigma=0.1)
  source = Source(spectrum, beam, rate=10.)
  timestamp, energy, x, y = source.rvs()

.. note ::

   At this point a :class:`Source <hexsample.source.Source>` is a very simple
   object describing a parallel beam of photons traveling along the z axis, that is,
   orthogonally to the detector plane---the latter will be always assumed to lie
   in the x-y plane---with no concept, e.g., of beam divergence. This is an area
   where we might want to add functionalities in the future.


Morphology
----------

The morphological part of the source description is encapsulated in a series of
subclasses of the (purely virtual) :class:`BeamBase <hexsample.source.BeamBase>`
class. More specifically:

* :class:`PointBeam <hexsample.source.PointBeam>` represents a point-like beam;
* :class:`DiskBeam <hexsample.source.DiskBeam>` represents a uniform disk;
* :class:`GaussianBeam <hexsample.source.GaussianBeam>` represents a simple gaussian beam.


Spectrum
--------

Likewise, the source spectrum is encapsulated in subclasses of the (virtual)
:class:`SpectrumBase <hexsample.source.SpectrumBase>` class and, particularly,
:class:`LineForest <hexsample.source.LineForest>`.


Module documentation
--------------------

.. automodule:: hexsample.source
