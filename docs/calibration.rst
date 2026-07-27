Detector calibration
====================

Detector calibration comprises two main aspects: the characterization of the
baseline and electronic noise, and the calibration of the pixel response. 


.. _pedestal-noise-calibration:

Pedestal and noise calibration
------------------------------

The pedestal and RMS noise are estimated independently for each detector pixel
and are expressed in ADC counts. Ideally, this calibration should be performed
using source-free data covering the entire detector. It can also be derived
from flat-field data acquired with a calibration source, such as
:math:`^{55}\mathrm{Fe}`, provided that pixels containing source signals are
excluded from the analysis.

For each accepted event, the noise samples are selected from the pixels inside
the region of interest (ROI) but outside the region of trigger (ROT). A
one-pixel-wide safety margin around the ROT is also excluded to reduce
contamination from signal charge:

.. code-block:: text

   +---------------------------+
   |  padding used for noise   |
   |  estimation              |
   |    +-----------------+    |
   |    | excluded margin |    |
   |    |  +-----------+  |    |
   |    |  |           |  |    |
   |    |  |  signal   |  |    |
   |    |  +-----------+  |    |
   |    +-----------------+    |
   |                           |
   +---------------------------+

Increasing the ROI padding generally increases the number of samples available
for each pixel and therefore improves the statistical precision of the
calibration.

The selected PHA samples are accumulated separately for each pixel. Their
sample mean provides an estimate of the pedestal, while their sample standard
deviation provides an estimate of the RMS noise. These quantities are measured
in ADC counts.

Once the pixel gain is known, the RMS noise can be converted into equivalent
noise charge (ENC), expressed in electrons, by dividing it by the gain in ADC
counts per electron.

.. _response-equalization:

Response equalization
---------------------

The pixel response is calibrated using data acquired with a source of known
energy spectrum, such as :math:`^{55}\mathrm{Fe}`. This procedure consists of
two stages: pixel equalization followed by gain calibration.

Equalization accounts for pixel-to-pixel variations in detector response. Two
methods are available:

* ``relative`` equalization determines the response of each pixel relative to
  the detector-wide average. The resulting factors are dimensionless and do
  not establish an absolute conversion between ADC counts and deposited
  energy;
* ``absolute`` equalization uses the known source spectrum to determine both
  the pixel-to-pixel response variations and the global ADC-to-energy
  conversion factor.

Consequently, a gain calibration with a physically meaningful absolute scale
can only be derived from the results of an ``absolute`` equalization.


