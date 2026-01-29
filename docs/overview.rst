.. _overview:

Overview
========

``hexsample`` is a Python package for the simulation and analysis of solid-state
hybrid detectors with hexagonal pixel sampling, providing the necessary
facilities to simulate, reconstruct and display data.

Command-line interface
----------------------

Once you have installed the package, the command-line interface provides access
to all the facilities at the most basic level.

.. program-output:: hexsample --help
   :prompt:

(You can also try ``hexsample <subcommand> --help`` to get help on specific
sub-commands.)


Tasks and pipelines
-------------------

At a slightly different level, the package is organized around the number of
high-level tasks packaged in the :mod:`~hexsample.tasks` module. These include:

* :meth:`~hexsample.tasks.simulate`: simulate detector data from a source;
* :meth:`~hexsample.tasks.reconstruct`: reconstruct detector data;
* :meth:`~hexsample.tasks.display`: event display;
* :meth:`~hexsample.tasks.quicklook`: quick-look analysis.

(You might have noticed that these tasks correspond to the sub-commands of the
command-line interface.)

Each task is then wrapped, with an identical name, in the
:mod:`~hexsample.pipeline` module.

.. seealso::
   :mod:`~hexsample.tasks`, :mod:`~hexsample.pipeline`

The main difference between the two is that all the functions in the
:mod:`~hexsample.tasks` module come with a precise, documented signature, while
the corresponding functions in the :mod:`~hexsample.pipeline` module are
entirely driven by keyword arguments. The basic rule is: if you are building
a simulation and/or analysis pipeline programmatically, always use the
functions in the :mod:`~hexsample.tasks` module; this will prevents typos in the
name of the arguments running undetected. The functions in the
:mod:`~hexsample.pipeline` module, on the other hand, are designed to filter
the keyword arguments and will happily ignore any unknown argument; they are
really only meant to be used by the command-line interface, since the validation
there happens at the ``argparse`` level.