.. _release_notes:

Release notes
=============


* Minor tweaks to the docs to make clear that all the lengths are in cm.
* Minor formatting changes.


*hexsample (0.8.1) - Tue, 12 Dec 2023 15:10:55 +0100*

* Merging https://github.com/lucabaldini/hexsample/pull/45
* Better parameter initialization for the DoubleGaussian model.
* Issue(s) closed:
      * https://github.com/lucabaldini/hexsample/issues/44


*hexsample (0.8.0) - Thu, 07 Dec 2023 12:24:46 +0100*

* Merging https://github.com/lucabaldini/hexsample/pull/42
* New functions in the analysis module.
* New ``scripts`` foldes, and first script to analyze the output of a
  thickness-noise scan.


*hexsample (0.7.0) - Wed, 25 Oct 2023 14:29:27 +0200*

* Merging https://github.com/lucabaldini/hexsample/pull/39
* Merging https://github.com/lucabaldini/hexsample/pull/38
* Merging https://github.com/lucabaldini/hexsample/pull/35
* Major refactoring of the modeling framework, with no (intentional) modification
  to the public API.
* New FitStatus class, refactoring of the FitModelBase class, with fit() and
  fit_histogram() now class members (as opposed to loose functions in the
  fitting module).
* Fit with parameter bounds now supported.
* Specific class for a double gaussian fit, with sensible initial values.


*hexsample (0.6.0) - Thu, 19 Oct 2023 23:23:07 +0200*

* Merging https://github.com/lucabaldini/hexsample/pull/35
* New random number generation scheme, (sort of) following the best practices
  suggested on the numpy documentation.
* Issue(s) closed:
      * https://github.com/lucabaldini/hexsample/issues/24


*hexsample (0.5.2) - Wed, 18 Oct 2023 21:45:47 +0200*

* New attempt at compiling the docs on github pages.


*hexsample (0.5.1) - Wed, 18 Oct 2023 21:30:42 +0200*

* Merging https://github.com/lucabaldini/hexsample/pull/3
* Attempt at compiling the docs on github pages whenever a new tag is created.


*hexsample (0.5.0) - Tue, 17 Oct 2023 22:37:49 +0200*

* Merging https://github.com/lucabaldini/hexsample/pull/29
* Merging https://github.com/lucabaldini/hexsample/pull/30
* Merging https://github.com/lucabaldini/hexsample/pull/31\
* Readout chip geometry exposed in the simulation via command-line arguments, and
  automatically picked up in the reconstruction and the event display.
* Start message updated.
* Bookkeeping in place for the file types.
* New "filetype" attribute added to the file header---written automatically by
  OutputFileBase and read automatically by InputFileBase.
* New fileio.open_input_file() function added to open input files transparently.
* Issue(s) closed:
      * https://github.com/lucabaldini/hexsample/issues/21
      * https://github.com/lucabaldini/hexsample/issues/27


*hexsample (0.4.0) - Mon, 16 Oct 2023 22:11:44 +0200*

* Merging https://github.com/lucabaldini/hexsample/pull/22
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
* Issue(s) closed:
      * https://github.com/lucabaldini/hexsample/issues/12


*hexsample (0.3.2) - Mon, 16 Oct 2023 12:12:10 +0200*

* Merging https://github.com/lucabaldini/hexsample/pull/25
* Command-line switch to set the random seed added.
* Version and tag date added to the output file header.
* Issue(s) closed:
      * https://github.com/lucabaldini/hexsample/issues/23


*hexsample (0.3.1) - Fri, 13 Oct 2023 15:41:01 +0200*

* Merging https://github.com/lucabaldini/hexsample/pull/20
* Added support for Python 3.7 through small tweaks to the type annotations.
* Added setup.bat script to support development under Windows.


*hexsample (0.3.0) - Fri, 13 Oct 2023 14:28:53 +0200*

* Merging https://github.com/lucabaldini/hexsample/pull/18
* Merging https://github.com/lucabaldini/hexsample/pull/19
* Cleanup and linting.
* Glaring bug in the simulation (the z coordinate of absorption was swapped) fixed.


*hexsample (0.2.0) - Thu, 12 Oct 2023 17:51:13 +0200*

* Merging https://github.com/lucabaldini/hexsample/pull/11
* Merging https://github.com/lucabaldini/hexsample/pull/10
* Merging https://github.com/lucabaldini/hexsample/pull/17
* Casting the outputfile default argument to string in ArgumentParser in order
  to avoid possible problems downstream with patlib.Path instances.
* mc option removed from output digi and recon files.
* Base classes for input and output files added, and machinery for adding
  and retrieving metadata information to/from file headers added.
* Digi header group metadata propagated to the recon files.
* io module renamed as fileio
* Added protection against mistyped parameter names in pipeline calls.
* uncertainties added as a requirement.
* PlotCard class completely refactored.
* Updating the hxview script.
* Issue(s) closed:
      * https://github.com/lucabaldini/hexsample/issues/14
      * https://github.com/lucabaldini/hexsample/issues/15


*hexsample (0.1.0) - Tue, 10 Oct 2023 10:31:12 +0200*

* Merging https://github.com/lucabaldini/hexsample/pull/10
* Initial setup of the repository.
* Simple versioning system in plac
