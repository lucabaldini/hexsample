# Copyright (C) 2022, luca.baldini@pi.infn.it
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""Advanced fitting facilities.
"""

import numpy as np

import aptapy.models

# pylint: disable=invalid-name

def fit_gaussian_iterative(histogram, p0=None, xmin=-np.inf, xmax=np.inf, absolute_sigma=True,
    check_finite=True, method=None, verbose=True, num_sigma_left=2., num_sigma_right=2.,
    num_iterations=2, **kwargs):
    """Fit the core of a gaussian histogram within a given number of sigma
    around the peak.

    This function performs a first round of fit to the data and then
    repeats the fit iteratively limiting the fit range to a specified
    interval defined in terms of deviations (in sigma) around the peak.

    For additional parameters look at the documentation of the
    :meth:`ixpeobssim.core.fitting.fit_histogram`

    Parameters
    ----------
    num_sigma_left : float
        The number of sigma on the left of the peak to be used to define the
        fitting range.

    num_sigma_right : float
        The number of sigma on the right of the peak to be used to define the
        fitting range.

    num_iterations : int
        The number of iterations of the fit.
    """
    # pylint: disable=too-many-arguments. too-many-locals
    model = aptapy.models.Gaussian()
    model.fit_histogram(histogram, p0, xmin=xmin, xmax=xmax, absolute_sigma=absolute_sigma, **kwargs)
    for i in range(num_iterations):
        xmin = model.mean.value - num_sigma_left * model.sigma.value
        xmax = model.mean.value + num_sigma_right * model.sigma.value
        try:
            model.fit_histogram(histogram, p0, xmin=xmin, xmax=xmax, absolute_sigma=absolute_sigma, **kwargs)
        except RuntimeError as exception:
            raise RuntimeError(f'Exception after {i + 1} iteration(s)') from exception
    return model
