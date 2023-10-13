.. _release_notes:

Release notes
=============


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