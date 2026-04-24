.. _calibration:

Calibration
===========

Two things: noise and gain.


Goal: two matrices with one number per pixel. Separate products in different files.


Calibration matrices
--------------------


Producing calibration files
---------------------------


Pedestal and noise
~~~~~~~~~~~~~~~~~~

Noise: we need a dataset, ideally with full illumination, with rectangular
readout. This is important because we are throwing away the pixels with signals
and using the padding region for measuring the noise.

Basic algorithm: take the pixel the highest value, remove it along with the 8
pixels around (in a rectangular sense, which makes the things easier to
implement and reason about with numpy arrays).

The basic assumption here is that the noise is gaussian with zero average, and
we are clipping the negative part with the zero suppression. The algorithm
is accumulating the sum of the squares, which works out of the box.

-> Issue for the new algorithm for pedestal and noise.

.. code-block:: shell

   hexsample calibrate noise input_file

This will produce a HDF5 file with two matrices, one with the noise values,
and one with the number of hits per pixels.

If there are no hits for a specific pixel, we just use the average of the pixels
with hits.


Gain
~~~~

Two-pass processes. We need a monochromatic source, so that we know the
average number of electrons/holes produced.

The process is based on a chisquare that can be minimized with respect to the
gain. The effect of the threshold introduces a bias, which can be corrected with
a Monte Carlo simulation that is run on the fly.


Using calibration files
-----------------------

