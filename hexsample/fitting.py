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


from loguru import logger
import numpy
from scipy.optimize import curve_fit

import hexsample.modeling

# pylint: disable=invalid-name


USE_ABSOLUTE_SIGMA = True


def fit(model, xdata, ydata, p0=None, sigma=None, xmin=-numpy.inf,
        xmax=numpy.inf, absolute_sigma=USE_ABSOLUTE_SIGMA, check_finite=True,
        method=None, verbose=True, **kwargs):
    """Lightweight wrapper over the ``scipy.optimize.curve_fit()`` function
    to take advantage of the modeling facilities. More specifically, in addition
    to performing the actual fit, we update all the model parameters so that,
    after the fact, we do have a complete picture of the fit outcome.

    Parameters
    ----------
    model : :py:class:`modeling.FitModelBase` instance callable
        The model function, f(x, ...).  It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.

    xdata : array_like
        The independent variable where the data is measured.

    ydata : array_like
        The dependent data --- nominally f(xdata, ...)

    p0 : None, scalar, or sequence, optional
        Initial guess for the parameters. If None, then the initial
        values will all be 1.

    sigma : None or array_like, optional
        Uncertainties in `ydata`. If None, all the uncertainties are set to
        1 and the fit becomes effectively unweighted.

    xmin : float
        The minimum value for the input x-values.

    xmax : float
        The maximum value for the input x-values.

    absolute_sigma : bool, optional
        If True, `sigma` is used in an absolute sense and the estimated
        parameter covariance `pcov` reflects these absolute values.
        If False, only the relative magnitudes of the `sigma` values matter.
        The returned parameter covariance matrix `pcov` is based on scaling
        `sigma` by a constant factor. This constant is set by demanding that the
        reduced `chisq` for the optimal parameters `popt` when using the
        *scaled* `sigma` equals unity.

    method : {'lm', 'trf', 'dogbox'}, optional
        Method to use for optimization.  See `least_squares` for more details.
        Default is 'lm' for unconstrained problems and 'trf' if `bounds` are
        provided. The method 'lm' won't work when the number of observations
        is less than the number of variables, use 'trf' or 'dogbox' in this
        case.

    verbose : bool
        Print the model if True.

    kwargs
        Keyword arguments passed to `leastsq` for ``method='lm'`` or
        `least_squares` otherwise.
    """
    # Select data based on the x-axis range passed as an argument.
    _mask = numpy.logical_and(xdata >= xmin, xdata <= xmax)
    xdata = xdata[_mask]
    ydata = ydata[_mask]
    if len(xdata) <= len(model.parameters):
        raise RuntimeError('Not enough data to fit (%d points)' % len(xdata))
    if isinstance(sigma, numpy.ndarray):
        sigma = sigma[_mask]
    # If the model has a Jacobian defined, go ahead and use it.
    try:
        jac = model.jacobian
    except:
        jac = None
    # If we are not passing default starting points for the model parameters,
    # try and do something sensible.
    if p0 is None:
        model.init_parameters(xdata, ydata, sigma)
        p0 = model.parameters
        if verbose:
            logger.debug('%s parameters initialized to %s.', model.name(), p0)
    # If sigma is None, assume all the errors are 1. (If we don't do this,
    # the code will crash when calculating the chisquare.
    if sigma is None:
        sigma = numpy.full((len(ydata), ), 1.)
    popt, pcov = curve_fit(model, xdata, ydata, p0, sigma, absolute_sigma,
        check_finite, model.bounds, method, jac, **kwargs)
    # Update the model parameters.
    model.set_plotting_range(xdata.min(), xdata.max())
    model.parameters = popt
    model.covariance_matrix = pcov
    model.chisq = (((ydata - model(xdata))/sigma)**2).sum()
    model.ndof = len(ydata) - len(model)
    if verbose:
        print(model)
    return model


def fit_histogram(model, histogram, p0=None, sigma=None, xmin=-numpy.inf,
                  xmax=numpy.inf, absolute_sigma=USE_ABSOLUTE_SIGMA,
                  check_finite=True, method=None, verbose=True, **kwargs):
    """Fit a histogram to a given model.

    This is basically calling :meth:`ixpeobssim.core.fitting.fit` with some
    pre-processing to turn the histogram bin edges and content into
    x-y data. Particularly, the bin centers are taken as the independent
    data series, the bin contents are taken as the dependent data saries,
    and the square root of the counts as the Poisson error.

    For additional parameters look at the documentation of the
    :meth:`ixpeobssim.core.fitting.fit`

    Parameters
    ----------
    model : :py:class:`modeling.FitModelBase` instance or
        callable
        The fit model.

    histogram : ixpeHistogram1d instance
        The histogram to be fitted.

    Warning
    -------
    We're not quite doing the right thing, here, as we should integrate
    the model within each histogram bin and compare that to the counts,
    but this is not an unreasonable first-order approximation. We might want
    to revise this, especially since we can probably provide an analytic
    integral for most of the model we need.
    """
    assert histogram.num_axes == 1
    _mask = (histogram.content > 0)
    xdata = histogram.bin_centers(0)[_mask]
    ydata = histogram.content[_mask]
    if sigma is None:
        sigma = numpy.sqrt(ydata)
    return fit(model, xdata, ydata, p0, sigma, xmin, xmax, absolute_sigma,
               check_finite, method, verbose, **kwargs)


def fit_gaussian_iterative(histogram, p0=None, sigma=None, xmin=-numpy.inf,
                           xmax=numpy.inf, absolute_sigma=USE_ABSOLUTE_SIGMA,
                           check_finite=True, method=None, verbose=True,
                           num_sigma_left=2., num_sigma_right=2.,
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
    fit_histogram(model, histogram, p0, sigma, xmin, xmax, absolute_sigma,
                  check_finite, method, verbose, **kwargs)
    for i in range(num_iterations):
        xmin = model.peak - num_sigma_left * model.sigma
        xmax = model.peak + num_sigma_right * model.sigma
        try:
            fit_histogram(model, histogram, p0, sigma, xmin, xmax,
                          absolute_sigma, check_finite, method, verbose,
                          **kwargs)
        except RuntimeError as e:
            raise RuntimeError('%s after %d iteration(s)' % (e, i + 1))
    return model
