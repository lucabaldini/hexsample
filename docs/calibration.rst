:mod:`~hexsample.calibration` 
==========================================

The :mod:`hexsample.calibration` module provides the data structures and algorithms required to calibrate the detector. It defines a common format for representing calibration results and provides facilities for saving calibration data to, and loading them from, HDF5 files.

The module contains fourteen classes that can be divided into three main categories. 

1) Enumerations and metadata
-------------------------

These classes define a common vocabulary for describing calibration products, including the calibration type, the associated metadata, and the corresponding units of measurement. They do not perform any calibration themselves.

* ``CalibrationType``
* ``CalibrationMetadata``
* ``PositionCalibrationMetadata``
* ``CalibrationUnits``

2) Calibration result containers
-----------------------------

These classes define how calibration results are represented and stored.

* ``CalibrationBase``
* ``CalibrationMatrix``
* ``PositionCalibrationData``

Most pixel-based calibrations use ``CalibrationMatrix``, which organizes the calibration values, uncertainties, and number of entries into arrays matching the detector geometry. Position-reconstruction calibration is an exception and uses ``PositionCalibrationData``, since its results cannot be represented by a single value for each detector pixel.

Both data containers inherit from ``CalibrationBase``, which provides common functionality for storing calibration values and metadata, as well as for serializing calibration data to and from HDF5 files.


3) Calibration algorithms
----------------------
These classes implement the procedures used to produce calibration results. 

* ``CalibrateBase``
* ``CalibrateNoise``
* ``CalibrateDark``
* ``CalibrateEqualization``
* ``CalibrateGain``
* ``CalibrateENC``
* ``CalibratePosition``

Depending on the type of calibration, they either analyze detector events and progressively accumulate information (``CalibrateNoise``, ``CalibrateDark``, ``CalibrateEqualization``, and ``CalibratePosition``) or combine the results of previous calibrations (``CalibrateGain`` and ``CalibrateENC``).


CalibrateBase
~~~~~~~~~~~~~
``CalibrateBase`` is the base class for calibration algorithms that produce a ``CalibrationMatrix``. It initializes the matrix in which derived calibration algorithms store the values, uncertainties, and number of entries calculated
for each detector pixel.


CalibrateNoise
~~~~~~~~~~~~~
``CalibrateNoise`` estimates the noise as the RMS of the PHA of each pixel, expressed in ADC counts. The calibration procedure is divided into two main stages: ``analyze_event()``, which selects and accumulates data on an event-by-event basis, and ``fit()``, which computes the final noise matrix.

For each accepted event, the pixels used to estimate the noise are selected from the ROI. To prevent the signal from contaminating the noise estimate, ``_remove_signal()`` excludes both the pixels in the ROT and a one-pixel-wide
safety margin around it:

.. code-block:: text

   +---------------------------+
   |  padding used             |
   |  for noise estimation     |
   |    +-----------------+    |
   |    | excluded margin |    |
   |    |  +-----------+  |    |
   |    |  |           |  |    |
   |    |  |  signal   |  |    |
   |    |  +-----------+  |    |
   |    +-----------------+    |
   |                           |
   +---------------------------+

Increasing the padding generally increases the number of samples available for each pixel and therefore improves the statistical precision of the noise estimate.

The squared PHA values of the selected pixels are then accumulated, together with the number of available samples for each pixel. Finally, ``fit()`` computes the RMS noise as the square root of the mean squared PHA for every pixel with at least one valid sample.


CalibrateDark
~~~~~~~~~~~~~
``CalibrateDark`` estimates both the pedestal and the RMS noise of each pixel, expressed in ADC counts. Ideally, the calibration should be performed using a source-free dataset acquired over the entire readout chip. If the dataset contains a source signal, contamination is reduced with the same masking strategy used by ``CalibrateNoise``: the ROT and a one-pixel-wide margin around it are excluded, leaving only the outer pixels of the ROI for the calibration.

As in ``CalibrateNoise``, the calibration procedure consists of two stages. ``analyze_event()`` accumulates the PHA samples for each pixel. ``fit()`` then produces and returns the noise and pedestal calibration matrices. Two algorithms are available:

* ``welford`` updates the sample count, mean, and variance of the PHA values online, without storing the individual samples. This implementation always applies the signal mask. The mean provides the pedestal estimate, while the sample standard deviation provides the RMS noise.
* ``fit`` builds the PHA distribution of each pixel and fits it with a Gaussian model. The fitted Gaussian mean is used as the pedestal, while its standard deviation is used as the RMS noise.


.. warning::

   The two algorithms currently handle ``has_source`` differently. The ``fit`` algorithm masks the ROT and its one-pixel margin only when ``has_source`` is true, whereas ``welford`` always applies this mask, regardless of ``has_source``.


CalibrateEqualization
~~~~~~~~~~~~~
``CalibrateEqualization`` estimates the pixel-to-pixel response variations of the detector using preprocessed event clusters. Each event is passed to ``analyze_cluster()`` as a ``Cluster`` object containing the coordinates and PHA values of the pixels identified as belonging to the signal. This allows the equalization algorithms to operate on data for which the pedestal has already been subtracted and a noise-dependent zero-suppression threshold has already been applied.

Two equalization algorithms are available: ``relative`` and ``absolute``. 
* The ``relative`` algorithm measures relative pixel-to-pixel response variations using only single-pixel clusters, for which the signal is assumed to be entirely contained within one pixel. For each pixel, it calculates the mean PHA of the selected events. These pixel means are then normalized by their detector-wide average, producing equalization values with a mean of one over the calibrated pixels.
* The ``absolute`` algorithm uses both single-pixel and multi-pixel clusters. It performs a likelihood fit that compares the equalized total PHA of each event with the known probability density function of the calibration-source spectrum. The equalization factors of multiple pixels are fitted simultaneously within overlapping regions of the detector. Unlike the ``relative`` algorithm, it can therefore use charge-sharing events and determine the global conversion factor from ADC counts to energy. This method requires a ``SpectrumPDF`` describing the source spectrum and is computationally more demanding.

.. note::

   Some of the functions used by the ``absolute`` equalization algorithm are defined at module level rather than as methods of ``CalibrateEqualization``. This allows the block fits to be serialized and executed in parallel. 

CalibrateGain
~~~~~~~~~~~~~
``CalibrateGain`` converts an absolute equalization matrix into a pixel gain matrix, expressed in ADC counts per electron. It uses the global ADC-to-energy conversion factor stored in the equalization matrix and the mean ionization energy of the sensor material, defined in ``Sensor``. The latter represents the average energy required to produce one electron-hole pair. Together, these quantities provide the conversion between ADC counts and generated charge, which is combined with the pixel equalization factors to determine the gain of each pixel.

.. warning::

   The gain has a physically meaningful absolute scale only when the input matrix was produced using the ``absolute`` equalization algorithm. 


CalibrateENC
~~~~~~~~~~~~
``CalibrateENC`` calculate the equivalent noise charge of each pixel from the noise and gain calibration matrices. The ENC, expressed in electrons, is obtained as the ratio between the RMS noise, expressed in ADC counts, and the pixel gain, expressed in ADC counts per electron. 



CalibratePosition
~~~~~~~~~~~~~


Module documentation
--------------------

.. automodule:: hexsample.calibration
