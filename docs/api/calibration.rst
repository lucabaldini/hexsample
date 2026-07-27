:mod:`~hexsample.calibration` --- Calibration
=============================================

The :mod:`hexsample.calibration` module provides the data structures and
algorithms required to calibrate the detector. It defines a common format for
representing calibration results and provides facilities for saving calibration
data to, and loading them from, HDF5 files.

The module contains fourteen classes that can be divided into three main
categories.

1) Enumerations and metadata
----------------------------

These classes define a common vocabulary for describing calibration products,
including the calibration type, the associated metadata, and the corresponding
units of measurement. They do not perform any calibration themselves.

* :class:`~hexsample.calibration.CalibrationType`
* :class:`~hexsample.calibration.CalibrationMetadata`
* :class:`~hexsample.calibration.PositionCalibrationMetadata`
* :class:`~hexsample.calibration.CalibrationUnits`

2) Calibration result containers
--------------------------------

These classes define how calibration results are represented and stored.

* :class:`~hexsample.calibration.CalibrationBase`
* :class:`~hexsample.calibration.CalibrationMatrix`
* :class:`~hexsample.calibration.PositionCalibrationData`

Most pixel-based calibrations use ``CalibrationMatrix``, which organizes the
calibration values, uncertainties, and number of entries into arrays matching
the detector geometry. Position-reconstruction calibration is an exception and
uses ``PositionCalibrationData``, since its results cannot be represented by a
single value for each detector pixel.

Both data containers inherit from ``CalibrationBase``, which provides common
functionality for storing calibration values and metadata, as well as for
serializing calibration data to and from HDF5 files.

3) Calibration algorithms
-------------------------

These classes implement the procedures used to produce calibration results.

* :class:`~hexsample.calibration.CalibrateBase`
* :class:`~hexsample.calibration.CalibrateNoise`
* :class:`~hexsample.calibration.CalibrateDark`
* :class:`~hexsample.calibration.CalibrateEqualization`
* :class:`~hexsample.calibration.CalibrateGain`
* :class:`~hexsample.calibration.CalibrateENC`
* :class:`~hexsample.calibration.CalibratePosition`

Depending on the type of calibration, they either analyze detector events and
progressively accumulate information (``CalibrateNoise``, ``CalibrateDark``,
``CalibrateEqualization``, and ``CalibratePosition``) or combine the results of
previous calibrations (``CalibrateGain`` and ``CalibrateENC``).

CalibrateBase
~~~~~~~~~~~~~

``CalibrateBase`` is the base class for calibration algorithms that produce a
``CalibrationMatrix``. It initializes the matrix in which derived calibration
algorithms store the values, uncertainties, and number of entries calculated
for each detector pixel.

CalibrateNoise
~~~~~~~~~~~~~~

``CalibrateNoise`` estimates the noise as the RMS of the PHA of each pixel,
expressed in ADC counts. The calibration procedure is divided into two main
stages: ``analyze_event()``, which selects and accumulates data on an
event-by-event basis, and ``fit()``, which computes the final noise matrix. For
each pixel, the RMS noise is calculated as the square root of the mean squared
PHA.

CalibrateDark
~~~~~~~~~~~~~

``CalibrateDark`` produces pedestal and RMS-noise calibration matrices from
detector events. Samples are accumulated with ``analyze_event()``, while
``fit()`` computes and returns the resulting calibration matrices.

Two algorithms are available:

* ``welford`` computes the sample mean and variance online;
* ``fit`` builds the PHA distribution of each pixel and fits it with a
  Gaussian model.

For a description of the event-selection and signal-masking procedure, see
:ref:`pedestal-noise-calibration`.

.. warning::

   The two algorithms currently handle ``has_source`` differently. The ``fit``
   algorithm masks the ROT and its one-pixel margin only when ``has_source`` is
   true, whereas ``welford`` always applies this mask, regardless of
   ``has_source``.

CalibrateEqualization
~~~~~~~~~~~~~~~~~~~~~

``CalibrateEqualization`` derives pixel equalization factors from preprocessed
event clusters. Each cluster is passed to ``analyze_cluster()`` as a
``Cluster`` object containing the coordinates and PHA values of the signal
pixels. The input data are expected to have undergone pedestal subtraction and
noise-dependent zero suppression.

Two algorithms are available:

* ``relative`` uses single-pixel clusters to estimate the response of each
  pixel relative to the detector-wide average. It produces dimensionless
  equalization factors normalized to have an average value of one over the
  calibrated pixels.

* ``absolute`` performs a likelihood fit using single-pixel and charge-sharing
  clusters. It simultaneously estimates the pixel equalization factors and
  the global conversion between ADC counts and energy. This algorithm requires
  a ``SpectrumPDF`` describing the calibration-source spectrum.

For a detailed description of the two equalization procedures, see
:ref:`response-equalization`.

.. note::

   The block-fitting functions used by the ``absolute`` algorithm are defined
   at module level so that individual fits can be serialized and executed in
   parallel.

CalibrateGain
~~~~~~~~~~~~~

``CalibrateGain`` converts an absolute equalization matrix into a pixel gain
matrix, expressed in ADC counts per electron. It uses the global ADC-to-energy
conversion factor stored in the equalization matrix and the mean ionization
energy of the sensor material, defined in ``Sensor``. The latter represents the
average energy required to produce one electron-hole pair. Together, these
quantities provide the conversion between ADC counts and generated charge,
which is combined with the pixel equalization factors to determine the gain of
each pixel.

.. warning::

   The gain has a physically meaningful absolute scale only when the input
   matrix was produced using the ``absolute`` equalization algorithm.

CalibrateENC
~~~~~~~~~~~~

``CalibrateENC`` calculates the equivalent noise charge of each pixel from the
noise and gain calibration matrices. The ENC, expressed in electrons, is
obtained as the ratio between the RMS noise, expressed in ADC counts, and the
pixel gain, expressed in ADC counts per electron.

CalibratePosition
~~~~~~~~~~~~~~~~~

Module documentation
--------------------

.. automodule:: hexsample.calibration
   :members:
   :show-inheritance: