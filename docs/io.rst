:mod:`hexsample.io` --- Input/Output
====================================

This module contains all the I/O related facilities, that is, the basic definition
of the file format and the proper classes to create and read back data files.

.. warning::
   We have made the provisional decision to have all the I/O implemented in the
   HDF5 format, which seems to be widely used and to support all the basic
   features that we need. That said, there seems to be two actively maintained
   Python interfaces to the HDF5 standard, implemented according to radically ]
   different design principle, namely:

   * `pytables <https://www.pytables.org/index.html>`_;
   * `h5py <https://www.h5py.org/>`_.

   It is not trivial for a non expert to really understand which one is more
   suited to our purposes, and we opted for ``pytables``, but this is a part of
   the package that might see drastic changes in the future.


Digitized data
--------------

The basic content of a digitized event contains all the event-by-event data that
would ordinarily be written out by the DAQ, i.e., the trigger identifier, the
timestamp, and all the quantities that are necessary in order to uniquely identify
the region of interest:

.. literalinclude:: ../hexsample/io.py
   :pyobject: DigiDescription

In addition, the PHA content of the ROI (which is a variable-length array by its
very nature), is encapsulated in a separate ``VLArray`` object in the same
group holding the digi table.

For simulated data, digitized files contain an additional table encapsulating the
ground truth information for the event.

.. literalinclude:: ../hexsample/io.py
   :pyobject: MonteCarloDescription




Module documentation
--------------------

.. automodule:: hexsample.io
