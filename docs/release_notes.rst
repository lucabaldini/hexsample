.. _release_notes:

Release notes
=============

* New position reconstruction algorithm based on the likelihood of the charge
  diffusion model implemented.
* New module `likelihood.py` containing the code for the likelihood evaluation
  added.
* Unified position reconstruction algorithms calibration file and the command
  into `calibgen position`.
* New clustering algorithm `ClusteringHex` implemented. This clustering always
  contains seven pixels: the seed pixel and all its six neighbors.
* New module `position.py` containing the core implementation of the position
  reconstruction algorithms added. As a consequence, the `eta.py` module has
  been deleted.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/136
  - https://github.com/lucabaldini/hexsample/pull/106
  - https://github.com/lucabaldini/hexsample/issues/126


Version 0.16.3 (2026-05-14)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Refactored zero suppression threshold to be interpreted as number of standard deviations of
  the noise instead of a fixed number.
* Added check of the calibration matrices at the readout initialization to fill the empty
  pixels, in order to avoid problems in the reconstruction.
* New algorithm added to calibrate the pixel equalization factors. This algorithm calculates
  the gain equalization factors without any assumption on the source spectrum, by analyzing
  the energy distributions of one-pixel events and comparing them across the pixels.
* `calibgen equalization` now accepts the argument `algorithm` to choose between two different
  calibration algorithms: "relative", without using the source spectrum, and "absolute", that
  needs the source spectrum PDF to be passed as an argument.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/135
  - https://github.com/lucabaldini/hexsample/pull/134
  - https://github.com/lucabaldini/hexsample/issues/133
  - https://github.com/lucabaldini/hexsample/issues/125  


Version 0.16.2 (2026-05-12)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Package dependencies updated: `joblib` and `iminuit` added.
* New calibration matrix type `equalization` added in `caldb.py` to store the pixel equalization
  factors.
* New `CalibrateEqualization` class to calibrate the pixel equalization factors. This is
  performed with a likelihood fit using the source spectrum probability density function (PDF)
  and the observed energy of the events. Corresponding CLI command `calibgen equalization` added
  to generate equalization calibration files from event files.
* The `CalibrateGain` class has been refactored. Now it produces a gain calibration matrix from
  an equalization calibration matrix, by using the ADC to eV conversion and the chip sensor
  material. The gain matrix is expressed in ADC counts per electron. Corresponding CLI command
  `calibgen gain` refactored to reflect this change.
* Some CLI commands refactored to use the equalization calibration file instead of the gain
  calibration file (`reconstruct`, `display`, `calibgen eta`).
* New `pdf.py` module containing facilities to calculate the PDF of the source spectrum and store
  it in a `.npz` file.
* Added `calibspec` CLI command to create a source spectrum PDF file from a reconstructed event
  file. The PDF is saved as a `.npz` file.
* New `adc` method to calculate the total ADC counts of the event added to `ReconEvent`. Now a
  reconstructed file contains both the ADC counts and the energy in eV of the events (if the 
  conversion factor from ADC to eV is available in the equalization calibration file).
* New `energy` method to calculate the total energy of the event added to `Cluster`. The
  conversion factor from ADC to eV is passed by `ClusteringNN`.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/132
  - https://github.com/lucabaldini/hexsample/issues/129


Version 0.16.1 (2026-05-12)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* New `stats.py` module with some common statistical functions, most notably
  a class to accumulate the running statistics using the Welford algorithm.
* `calibgen dark` now accepts the argument `algorithm` to choose between two different algorithms
  to calibrate the noise and pedestal of the readout chip: "welford", to use Welford's online
  algorithm, that updates the mean and variance of the counts distribution on the fly, and "fit",
  that updates a 3D histogram of the counts distribution and fits it with a Gaussian to extract
  the noise and pedestal.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/131
  - https://github.com/lucabaldini/hexsample/issues/130


Version 0.16.0 (2026-05-05)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `display` command modified to plot just the events when no calibration files are provided,
  otherwise the reconstructed and Monte Carlo positions are calculated and plotted.
* Minor changes to `CalibrateDark` and `CalibrateNoise` signal removal.
* New `calibview` command to visualize calibration files and compare them with Montecarlo truth.
* New beam shapes added: `SquareBeam` and `RectangleBeam`.
* Fixed a bug in digitization of events outside and at the border of the readout chip. For
  rectangular readout, this problem is solved by masking out all the electrons that fall outside
  the chip during the readout. For circular readout, we also exclude events with the seed pixel
  at the border of the readout chip.
* New dataclasses in `xpol.py` to store properties of the readout chips (XPOL1 and XPOL3).
* Introduced the CalibrationDataBase (CalDB) to store and manage calibration files, along wit the
  dedicated module `caldb.py` to handle operations on the CalDB.
* New `calibgen synthesize` command to generate synthetic calibration files with user-defined
  properties for testing and simulation purposes.
* Refactoring of the calibration CLI commands, with the introduction of the `calibgen` command and
  the respective subcommands to produce calibration files from event files.
* Support for scalar properties in the readout chip (ENC, noise, pedestal, gain) has been removed
  in favor of the calibration files that store this information. All the CLI commands that need
  readout chip properties now require the corresponding calibration files to be passed as
  arguments.
* Added classes `CalibrateGain`, `CalibrateDark` and `CalibrateENC` to create gain, noise, pedestal
  and ENC calibration files.
* Refactoring of the calibration matrices classes, with the introduction of a new unified
  `CalibrationMatrix` class, and the deprecation of the old `CalibrationMatrixGain` and
  `CalibrationMatrixNoise` classes. This class is intended to store all the calibration information
  needed for every feature of the readout chip.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/128
  - https://github.com/lucabaldini/hexsample/pull/127
  - https://github.com/lucabaldini/hexsample/pull/124
  - https://github.com/lucabaldini/hexsample/pull/122
  - https://github.com/lucabaldini/hexsample/pull/121
  - https://github.com/lucabaldini/hexsample/pull/118
  - https://github.com/lucabaldini/hexsample/pull/111
  - https://github.com/lucabaldini/hexsample/pull/123
  - https://github.com/lucabaldini/hexsample/issues/120
  - https://github.com/lucabaldini/hexsample/issues/117
  - https://github.com/lucabaldini/hexsample/issues/116
  - https://github.com/lucabaldini/hexsample/issues/115
  - https://github.com/lucabaldini/hexsample/issues/113
  - https://github.com/lucabaldini/hexsample/issues/112
  - https://github.com/lucabaldini/hexsample/issues/99
  - https://github.com/lucabaldini/hexsample/issues/7


Version 0.15.0 (2026-04-23)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* All calibration scripts turned into tasks.
* Small refactoring of the tasks module.
* Parameters for the eta function reconstruction renamed and packed into a
  dictionary when passed around.
* Implemented `SlantedEdgeResolution` class to calculate the estimate the resolution of the detector
  using the slanted edge method. This class calculates the edge spread function (ESF), the line
  spread function (LSF) and the modulation transfer function (MTF) for a slit beam.
* Implemented `SlitsAligner` class to align a slit beam from a Huttner test.
* Added possibility to add an offset to the readout and subtract it during clustering, allowing
  to have negative pha values in the events.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/107
  - https://github.com/lucabaldini/hexsample/pull/105
  - https://github.com/lucabaldini/hexsample/pull/104
  - https://github.com/lucabaldini/hexsample/pull/102
  - https://github.com/lucabaldini/hexsample/issues/90
  - https://github.com/lucabaldini/hexsample/issues/89


Version 0.14.0 (2026-03-03)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added classes `CalibrationMatrixGain` and `CalibrationMatrixNoise` to use gain and noise
  matrices to support non-uniform pixel response in the readout chip.
* Command `calibrate` added to the CLI interface to use a DigiFile to calibrate the gain response
  and noise of the readout chip.
* Implemented the possibility to use the gain calibration matrix to simulate and reconstruct
  data with non-uniform pixel response, with the option `--map_gain_file` in the cli interface.
* `ClusteringBase` modified to accept a `HexagonalReadoutBase` instance instead of `HexagonalGrid`.
  This guarantees that the reconstruction can be performed even with non-uniform pixel gain.
* `Cluster` now has `col` and `row` arguments in the constructor, in order always carry the info
  about the logical coordinates of the event.
* New `legacy.py` module containing the code to convert .mdat3 files to .h5 files.
* New command `convert` to convert .mdat3 files to .h5 files using the CLI.
* Added an actual interactive event display with some GUI.
* Added support for random access in event files.
* Small cleanup in the fileio.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/103
  - https://github.com/lucabaldini/hexsample/pull/101
  - https://github.com/lucabaldini/hexsample/pull/100
  - https://github.com/lucabaldini/hexsample/issues/93
  - https://github.com/lucabaldini/hexsample/issues/67


Version 0.13.3 (2026-02-10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added rough support for DigiEventCircular objects in the event display.
* New `resolution.py` and `calibration.py` modules incorporating some common functions that were
  previously living in the script area.
* New script to calculate the EEF for events reconstructed with the
  different algorithms, disaggregated by cluster size.
* `--max_neighbors` option added to the cli interface.
* eta reconstructions modified to use the centroid for events with more than
  three pixels.
* Two parameters added to the eta reconstruction to control the transition from linear to probit
  reconstruction for two and three pixel events.
* New suppression step added to accept at most three pixels in a cluster, with the condition
  that all the pixels are neighbors.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/9
  - https://github.com/lucabaldini/hexsample/pull/94
  - https://github.com/lucabaldini/hexsample/pull/92


Version 0.13.2 (2026-02-04)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Modified the model to fit the angular coordinate for three-pixel events eta reconstruction.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/91


Version 0.13.1 (2026-01-30)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added scripts to calibrate the eta function.
* Added more command-line options to the cli interface to control the reconstruction
  with the eta function.
* Bug fix in the hexagonal beam implementation.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/82


Version 0.13.0 (2026-01-28)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* New command-line interface---this adds the new cli and tasks modules, and removes
  the old app module, as well as all the application scripts in the bin folder.
  This is the biggest single change in this release, and has a number of
  ramifications, with many files and modules affected.
* New base module, containing a few base classes and utilities for general use,
  including a TypeProxy class to handle dynamic typing.
* Major refactoring of the classes in the readout module, mainly for making the
  signature of the `read()` method consistent across all types.
* Refactoring of the source and sensor classes.
* Sparse readout and all associated data structures (including the event and
  file io machinery) removed, as deemed un-necessary.
* pprint module renamed as pretty to avoid confusion with the standard library
  pprint module.
* Limiting the pytest target in the ci workflow.
* Documentation and unit tests updated.
* General cleanup and linting, with removal of obsolete files such as test_optimize_sim.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/87
  - https://github.com/lucabaldini/hexsample/issues/79
  - https://github.com/lucabaldini/hexsample/issues/60
  - https://github.com/lucabaldini/hexsample/issues/52
  - https://github.com/lucabaldini/hexsample/issues/50


Version 0.12.0 (2025-12-20)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added the possibility to use the eta function to reconstruct events.
* Added new triangular and hexagonal (spatially) uniform beams.
* Added a new (spectrally) monochromatic beam.
* Obsolete fitting module removed.
* aptapy version bumped to >=0.18.0
* Release notes re-formatted and cleaned up.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/77
  - https://github.com/lucabaldini/hexsample/pull/76
  - https://github.com/lucabaldini/hexsample/issues/74
  - https://github.com/lucabaldini/hexsample/issues/73
  - https://github.com/lucabaldini/hexsample/issues/72
  - https://github.com/lucabaldini/hexsample/issues/69


Version 0.11.0 (2025-11-19)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Major repository restructuring to modernize the package layout and tooling infrastructure.
* Adoption of the "src" layout for the package.
* Migrate from a legacy setuptools-based structure to a modern pyproject.toml-based build system.
* Introduced nox for task automation (testing, linting, documentation building).
* Rewrote release tooling in tools/release.py with improved version management.
* Added pytest fixtures in tests/conftest.py for test data handling and matplotlib figure management.
* Consolidated logging configuration into `src/hexsample/logging_.py` module.
* Old setup files removed in favor of an editable install via pip.
* Old Makefile removed.
* Documentation sphinx theme changed.
* __init__.py file cleaned up.
* Full linting with ruff and pylint.
* github workflows updated.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/63
  - https://github.com/lucabaldini/hexsample/issues/59


Version 0.10.0 (2025-11-13)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Adding aptapy as a new dependency in both requirements.txt and pyproject.toml.
* Removing internal modules: hexsample/hist.py, hexsample/modeling.py, hexsample/plot.py
  and their test files.
* Updating imports across all test files, source files, and command-line scripts
  to use aptapy.hist, aptapy.models, and aptapy.plotting.
* Modified API calls to match aptapy's interface.
* Updates CI/CD configuration to use Ubuntu 22.04 and test on Python 3.7 and 3.13.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/58


Version 0.9.0 (2025-04-30)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Implemented circular and sparse readout modes.
* Minor tweaks to the docs to make clear that all the lengths are in cm.
* Minor formatting changes.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/46


Version 0.8.1 (2023-12-12)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Better parameter initialization for the DoubleGaussian model.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/45
  - https://github.com/lucabaldini/hexsample/issues/44


Version 0.8.0 (2023-12-07)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* New functions in the analysis module.
* New ``scripts`` folder, and first script to analyze the output of a
  thickness-noise scan.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/42


Version 0.7.0 (2023-10-25)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Major refactoring of the modeling framework, with no (intentional) modification
  to the public API.
* New FitStatus class, refactoring of the FitModelBase class, with fit() and
  fit_histogram() now class members (as opposed to loose functions in the
  fitting module).
* Fit with parameter bounds now supported.
* Specific class for a double gaussian fit, with sensible initial values.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/39
  - https://github.com/lucabaldini/hexsample/pull/38


Version 0.6.0 (2023-10-19)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* New random number generation scheme, (sort of) following the best practices
  suggested on the numpy documentation.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/35
  - https://github.com/lucabaldini/hexsample/issues/24


Version 0.5.2 (2023-10-18)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* New attempt at compiling the docs on github pages.


Version 0.5.1 (2023-10-18)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Attempt at compiling the docs on github pages whenever a new tag is created.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/3


Version 0.5.0 (2023-10-17)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Readout chip geometry exposed in the simulation via command-line arguments, and
  automatically picked up in the reconstruction and the event display.
* Start message updated.
* Bookkeeping in place for the file types.
* New "filetype" attribute added to the file header---written automatically by
  OutputFileBase and read automatically by InputFileBase.
* New fileio.open_input_file() function added to open input files transparently.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/31
  - https://github.com/lucabaldini/hexsample/pull/30
  - https://github.com/lucabaldini/hexsample/pull/29
  - https://github.com/lucabaldini/hexsample/issues/27
  - https://github.com/lucabaldini/hexsample/issues/21


Version 0.4.0 (2023-10-16)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* This is a major rework of the sampling, trigger and digitization facilities
  allowing for a simulation speedup of almost an order of magnitude, without loss
  of performance.
* Digitization machinery refactored in order to avoid working with large sparse
  arrays (in pixel and minicluster space) full of zeroes.
* Generation of the noise moved at the end of the digitization process.
* Hexagonal sampling largely rewritten to avoid the use of numpy.histogram2d.
* Trigger machinery reworked to accommodate the previous changes.
* Comparison operator defined for Padding, RegionOfInterest and DigiEvent in
  order to be able to make strict comparisons between output digi files.
* Seed for a small utility to compare digi files added.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/22
  - https://github.com/lucabaldini/hexsample/issues/12


Version 0.3.2 (2023-10-16)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Command-line switch to set the random seed added.
* Version and tag date added to the output file header.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/25
  - https://github.com/lucabaldini/hexsample/issues/23


Version 0.3.1 (2023-10-13)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added support for Python 3.7 through small tweaks to the type annotations.
* Added setup.bat script to support development under Windows.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/20


Version 0.3.0 (2023-10-13)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Cleanup and linting.
* Glaring bug in the simulation (the z coordinate of absorption was swapped) fixed.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/19
  - https://github.com/lucabaldini/hexsample/pull/18


Version 0.2.0 (2023-10-12)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Merging https://github.com/lucabaldini/hexsample/pull/11
* Merging https://github.com/lucabaldini/hexsample/pull/10
* Merging https://github.com/lucabaldini/hexsample/pull/17
* Casting the outputfile default argument to string in ArgumentParser in order
  to avoid possible problems downstream with pathlib.Path instances.
* mc option removed from output digi and recon files.
* Base classes for input and output files added, and machinery for adding
  and retrieving metadata information to/from file headers added.
* Digi header group metadata propagated to the recon files.
* io module renamed as fileio
* Added protection against mistyped parameter names in pipeline calls.
* uncertainties added as a requirement.
* PlotCard class completely refactored.
* Updating the hxview script.
* Pull requests merged and issues closed:
  - https://github.com/lucabaldini/hexsample/pull/17
  - https://github.com/lucabaldini/hexsample/pull/11
  - https://github.com/lucabaldini/hexsample/pull/10
  - https://github.com/lucabaldini/hexsample/issues/15
  - https://github.com/lucabaldini/hexsample/issues/14


Version 0.1.0 (2023-10-10)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Initial setup of the repository.
* Simple versioning system in place.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/hexsample/pull/10
