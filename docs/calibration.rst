Detector calibration
====================

Raw PHA values recorded by the detector must be corrected for electronic
baseline offsets (pedestal) and pixel-to-pixel gain variations (equalization)
to recover the deposited charge. Characterizing the pixel noise is also
essential for defining zero-suppression thresholds and correctly weighting PHA
values during clustering and event reconstruction.

Detector calibration accounts for spatial non-uniformities across the active
area. These may arise from small imperfections in the readout electronics,
sensor inhomogeneities, and variations in the bump-bonding connections to the
ASIC. Correcting them improves detector performance as well as spatial and
energy resolution.

The calibration process measures the pedestal level, the noise standard
deviation, and the normalized gain of each pixel in the matrix. The pedestal
and noise are expressed in ADC counts. During event reconstruction, the
resulting maps are used to subtract baseline offsets, set noise-dependent
thresholds, and equalize the pixel response. Under specific conditions, the
absolute gain and equivalent noise charge (ENC) can also be derived by
combining these calibration products with information about the incident
radiation.

In the analysis framework, a calibration product is stored as an HDF5 file
containing three matrices with the same shape as the readout chip. Each matrix
element corresponds to the pixel at the same position. The matrices are:

* ``values``: the central value of the calibrated quantity for each pixel;
* ``entries``: the number of events used to calibrate each pixel;
* ``errors``: the estimated uncertainty on the calibrated value for each
  pixel.

When calibration products are derived from a data file, the calibration steps
must be performed in a specific order. The pedestal and noise maps are
generated first, and the normalized gain map is then evaluated using these
baseline calibration products. If an absolute ADC-to-energy conversion is
available, the absolute gain and ENC maps can subsequently be derived from the
normalized gain and ADC noise maps.

Calibration products are stored as HDF5 files in the ``caldb`` directory. 
In addition to being used for event reconstruction, these
products can be supplied as input to detector simulations, allowing the
measured pixel response and electronic noise to be reproduced.

.. _pedestal-noise-calibration:

Pedestal and noise calibration
------------------------------

The pedestal and RMS noise are estimated independently for each detector pixel
and are expressed in ADC counts. In this framework, the calibration is derived
from data acquired with a source, excluding pixels that contain source signals
from the analysis.

For each accepted event, the noise samples are selected from the pixels inside
the ROI but outside the ROT. A one-pixel-wide safety margin around the ROT is 
also excluded to reduce contamination from signal charge:

.. code-block:: text

   +---------------------------+
   |  padding used for noise   |
   |  estimation               |
   |    +-----------------+    |
   |    | excluded margin |    |
   |    |  +-----------+  |    |
   |    |  |           |  |    |
   |    |  |  signal   |  |    |
   |    |  +-----------+  |    |
   |    +-----------------+    |
   |                           |
   +---------------------------+

The selected PHA samples are accumulated separately for each pixel. The
pedestal and RMS noise are identified with the mean and standard deviation of
the corresponding PHA distribution, respectively. These quantities can be
estimated either directly from the sample statistics, computed using Welford's
online algorithm, or from the mean and standard deviation of a Gaussian fit to
the distribution. Both the pedestal and RMS noise are expressed in ADC counts.

Increasing the ROI padding generally increases the number of samples available
for each pixel and therefore improves the statistical precision of the
calibration.

Once the pixel gain is known, the RMS noise can be eventually converted into ENC, 
expressed in electrons, by dividing it by the gain in ADC counts per electron.

.. _response-equalization:

Response equalization
---------------------

The pixel response is calibrated using data acquired by illuminating the
detector with a reference source, preferably monochromatic. Equalization
corrects pixel-to-pixel variations in detector response and improves both
energy and position reconstruction.

Two equalization methods are available:

* ``relative`` equalization uses single-pixel events and does not require prior
  knowledge of the source spectrum. The response of each pixel is estimated
  from the mean of its PHA distribution and normalized to the detector-wide
  average. The resulting correction factors are dimensionless and do not
  provide an absolute conversion between ADC counts and deposited energy;

* ``absolute`` equalization uses a model of the source spectrum and includes
  both single-pixel and multi-pixel events. The pixel correction factors are
  determined by maximizing the likelihood of the total corrected charge
  collected in each event. When the source spectrum is accurately modeled,
  this method also provides the global conversion factor between ADC counts
  and deposited energy.

The ``relative`` method requires fewer assumptions but uses only a subset of
the available events. The ``absolute`` method exploits a larger event sample,
but requires an accurate model of the source spectrum.

Consequently, a gain calibration with a physically meaningful absolute scale
can only be derived from an ``absolute`` equalization.