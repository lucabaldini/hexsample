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


import numpy
from scipy.optimize import curve_fit

from hexsample import logger
import hexsample.modeling

# pylint: disable=invalid-name


def fit_gaussian_iterative(histogram, p0=None, sigma=None, xmin=-numpy.inf,
    xmax=numpy.inf, absolute_sigma=True, check_finite=True,
    method=None, verbose=True, num_sigma_left=2., num_sigma_right=2.,
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
    model = hexsample.modeling.Gaussian()
    model.fit_histogram(histogram, p0, xmin, xmax, absolute_sigma, check_finite,
        method, verbose, **kwargs)
    for i in range(num_iterations):
        xmin = model.status['mean'] - num_sigma_left * model.status['sigma']
        xmax = model.status['mean'] + num_sigma_right * model.status['sigma']
        try:
            model.fit_histogram(histogram, p0, xmin, xmax, absolute_sigma,
                check_finite, method, verbose, **kwargs)
        except RuntimeError as e:
            raise RuntimeError('%s after %d iteration(s)' % (e, i + 1))
    return model
