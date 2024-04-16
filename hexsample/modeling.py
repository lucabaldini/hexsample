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

"""Fit models.
"""

from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit

from hexsample import logger
from hexsample.hist import Histogram1d
from hexsample.plot import plt, PlotCard, last_line_color

# pylint: disable=invalid-name


class FitStatus:

    """Small container class holding the fit status.

    Arguments
    ---------
    par_names : tuple of strings
        The names of the fit parameters.

    par_values : array_like
        The initial values of the fit parameters. This must an iterable of the same
        length of the parameter names, and is converted to a numpy array in the
        constructor.

    par_bounds : tuple, optional
        The initial bounds for the fit parameters. This is either None (in which ]
        case the bounds are assumed to be -np.inf--np.inf for all the fit parameters)
        or a 2-element tuple of iterables of the the same length of the parameter
        names expressing the minimum and maximum bound for each parameter.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, par_names: Tuple[str], par_values: np.ndarray,
        par_bounds: Tuple = None) -> None:
        """Constructor.
        """
        if len(par_names) != len(par_values):
            raise RuntimeError('Mismatch between parameter names and values')
        self.par_names = par_names
        self.par_values = np.array(par_values)
        self.num_params = len(self.par_names)
        self.par_bounds = self._process_bounds(par_bounds)
        self.par_covariance = np.zeros((self.num_params, self.num_params), dtype=float)
        self.chisquare = -1.
        self.ndof = -1
        self._index_dict = {name: i for i, name in enumerate(self.par_names)}

    def _process_bounds(self, par_bounds: Tuple = None) -> Tuple[np.ndarray, np.ndarray]:
        """Small utility functions to process the parameter bounds for later use.

        Verbatim from the scipy documentation, there are two ways to specify the bounds:

        * Instance of Bounds class.
        * 2-tuple of array_like: Each element of the tuple must be either an array
          with the length equal to the number of parameters, or a scalar (in which
          case the bound is taken to be the same for all parameters). Use np.inf with
          an appropriate sign to disable bounds on all or some parameters.

        Since we want to keep track of the bounds and, possibly, change them,
        we turn all allowed possibilities, here, into a 2-tuple of numpy arrays.
        """
        if par_bounds is None:
            return (np.full(self.num_params, -np.inf), np.full(self.num_params, np.inf))
        if len(par_bounds) != 2:
            raise RuntimeError(f'Invalid parameter bounds {par_bounds}')
        return tuple(np.array(bounds) for bounds in par_bounds)

    def _index(self, par_name: str) -> int:
        """Convenience method returning the index within the parameter vector
        for a given parameter name.
        """
        try:
            return self._index_dict[par_name]
        except KeyError as exception:
            raise KeyError(f'Unknown parameter "{par_name}"') from exception

    def parameter_value(self, par_name: str) -> float:
        """Return the parameter value for a given parameter indexed by name.

        Arguments
        ---------
        par_name : str
            The parameter name.
        """
        return self.par_values[self._index(par_name)]

    def __getitem__(self, par_name: str) -> float:
        """Convenience shortcut to retrieve the value of a parameter.

        Arguments
        ---------
        par_name : str
            The parameter name.
        """
        return self.parameter_value(par_name)

    def parameter_errors(self) -> np.ndarray:
        """Return the vector of parameter errors, that is, the square root of the
        diagonal elements of the covariance matrix.
        """
        return np.sqrt(self.par_covariance.diagonal())

    def parameter_error(self, par_name: str) -> float:
        """Return the parameter error by name.

        Arguments
        ---------
        par_name : str
            The parameter name.
        """
        index = self._index(par_name)
        return np.sqrt(self.par_covariance[index][index])

    def __iter__(self):
        """Iterate over the fit parameters as (name, value, error).
        """
        return zip(self.par_names, self.par_values, self.parameter_errors(), *self.par_bounds)

    def reduced_chisquare(self) -> float:
        """Return the reduced chisquare.
        """
        return self.chisquare / self.ndof if self.ndof > 0 else -1.

    def set_parameter_bounds(self, par_name: str, min_: float, max_: float) -> None:
        """Set the baounds for a given parameter.

        Arguments
        ---------
        par_name : str
            The parameter name.

        min_ : float
            The minimum bound.

        max_ : float
            The maximum bound.
        """
        index = self._index(par_name)
        self.par_bounds[0][index] = min_
        self.par_bounds[1][index] = max_

    def set_parameter(self, par_name: str, value: float) -> None:
        """Set the value for a given parameter (indexed by name).

        Arguments
        ---------
        par_name : str
            The parameter name.

        value : float
            The parameter value.
        """
        self.par_values[self._index(par_name)] = value

    def update(self, popt: np.ndarray, pcov: np.ndarray, chisq: float, ndof: int) -> None:
        """Update the data structure after a fit.

        Arguments
        ---------
        popt : array_like
            The array of best-fit parameters from scipy.optimize.curve_fit().

        pcov : array_like
            The covariance matrix of the paremeters from scipy.optimize.curve_fit().

        chisq : float
            The value of the chisquare.

        ndof : int
            The number of degrees of freedom from the fit.
        """
        self.par_values = popt
        self.par_covariance = pcov
        self.chisquare = chisq
        self.ndof = ndof

    def __str__(self):
        """String representation.
        """
        text = ''
        for name, value, error, min_bound, max_bound in self:
            text += f'{name:18s} {value:.3e} +/- {error:.3e} ({min_bound:.3e} / {max_bound:.3e})\n'
        text += f'Chisquare = {self.chisquare:.2f} / {self.ndof} dof'
        return text



class FitModelBase:

    """Base class for a fittable model.

    The class features a number of static members that derived class should redefine
    as needed:

    * ``PAR_NAMES`` is a list containing the names of the model parameters;
    * ``PAR_DEFAULT_VALUES`` is a list containing the default values
      of the model parameters;
    * ``PAR_DEFAULT_BOUNDS`` is a tuple containing the default values
      of the parameter bounds to be used for the fitting;
    * ``DEFAULT_RANGE`` is a two-element list with the default support
      (x-axis range) for the model. (This is automatically updated at runtime
      depending on the input data when the model is used in a fit.)

   In addition, each derived class should override the following methods:

    * the ``eval(x, *args)`` should return the value of the model at a given x
      for a given set of values of the underlying parameters;
    * the ``jacobian(x, *args)`` method, if defined, is passed to the underlying
      fit engine allowing to reduce the number of function calls in the fit.

    Finally, if there is a sensible way to initialize the model parameters
    based on a set of input data, derived classes should overload the
    ``init_parameters(xdata, ydata)`` method of the base class, as the
    latter is called by fitting routines if no explicit array of initial values
    are passed as an argument. The default behavior of the class method defined
    in the base class is to do nothing.

    See :class:`hexsample.modeling.Gaussian` for a working example.
    """

    # pylint: disable=too-many-instance-attributes

    PAR_NAMES = None
    PAR_DEFAULT_VALUES = None
    PAR_DEFAULT_BOUNDS = None
    DEFAULT_RANGE = (0., 1.)

    def __init__(self) -> None:
        """Constructor.
        """
        self.status = FitStatus(self.PAR_NAMES, self.PAR_DEFAULT_VALUES, self.PAR_DEFAULT_BOUNDS)
        self._xmin, self._xmax = self.DEFAULT_RANGE

    def name(self) -> str:
        """Return the model name.
        """
        return self.__class__.__name__

    def __getitem__(self, par_name: str) -> float:
        """Convenience shortcut to retrieve the value of a parameter.

        Arguments
        ---------
        par_name : str
            The parameter name.
        """
        return self.status.parameter_value(par_name)

    def set_parameter(self, par_name: str, value: float) -> None:
        """Convenience function to set the value for a given parameter in the
        underlying FitStatus object.
        """
        self.status.set_parameter(par_name, value)

    def set_range(self, xmin: float, xmax: float) -> None:
        """Set the function range.

        Arguments
        ---------
        xmin : float
            The minimum x-range value.

        xmax : float
            The maximum x-range value.
        """
        self._xmin = xmin
        self._xmax = xmax

    def plot(self, num_points: int = 250, **kwargs) -> None:
        """Plot the model.
        """
        x = np.linspace(self._xmin, self._xmax, num_points)
        plt.plot(x, self(x), **kwargs)

    def stat_box(self, x: float = 0.95, y: float = 0.95) -> None:
        """Plot a stat box for the model.
        """
        box = PlotCard()
        box.add_string('Fit model', self.name())
        box.add_string('Chisquare', f'{self.status.chisquare:.1f} / {self.status.ndof}')
        for name, value, error, *_ in self.status:
            box.add_quantity(f'{name.title()}', value, error)
        box.plot(x, y)

    def __call__(self, x, *parameters):
        """Return the value of the model at a given point and a given set of
        parameter values.

        The function accept a variable number of parameters, with the understanding that

        * if all the parameters are passed (that is, the number of arguments is the
          same as the number of the function parameters) then the function is
          evaluated in correspondence of those parameters;
        * if no parameter is passed, then the function is evaluated in correspondence
          of the ``self.parameters`` class member;
        * a ``RuntimeError`` is raised in all other cases.

        The function signature is designed so that the __call__ dunder can be
        called by ``curve_fit`` during the fitting process, and then we can reuse
        it to plot the model after the fit.
        """
        num_params = len(parameters)
        if num_params == self.status.num_params:
            return self.eval(x, *parameters)
        if num_params == 0:
            return self.eval(x, *self.status.par_values)
        raise RuntimeError(f'Wrong number of parameters ({num_params}) to {self.name()}.__call__()')

    @staticmethod
    def eval(x: np.ndarray, *parameters) -> np.ndarray:
        """Eval the model at a given x and a given set of parameter values.

        This needs to be overloaded by any derived classes.
        """
        raise NotImplementedError

    def init_parameters(self, xdata: np.ndarray, ydata: np.ndarray) -> None:
        """Assign a sensible set of values to the model parameters, based on a data
        set to be fitted.

        In the base class the method is not doing anything, but it can be reimplemented
        in derived classes to help make sure the fit converges without too much manual intervention.

        Arguments
        ---------
        xdata : array_like
            The x data.

        ydata : array_like
            The y data.
        """

    def fit(self, xdata: np.ndarray, ydata: np.ndarray, p0: np.ndarray = None,
        sigma: np.ndarray = None, xmin: float = -np.inf, xmax: float = np.inf,
        absolute_sigma: bool = True, check_finite: bool =True, method: str = None,
        verbose: bool = True, **kwargs):
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
        # pylint: disable=too-many-arguments, too-many-locals
        # If sigma is None, assume all the errors are 1---we need to do this
        # explicitely in order to calculate the chisquare downstream.
        if sigma is None:
            sigma = np.full(len(ydata), 1.)
        # Select data based on the x-axis range passed as an argument.
        mask = np.logical_and(xdata >= xmin, xdata <= xmax)
        xdata = xdata[mask]
        if len(xdata) <= self.status.num_params:
            raise RuntimeError(f'Not enough data to fit ({len(xdata)} points)')
        ydata = ydata[mask]
        sigma = sigma[mask]
        # If the model has a Jacobian defined, go ahead and use it.
        try:
            jac = self.jacobian
        except AttributeError:
            jac = None
        # If we are not passing default starting points for the model parameters,
        # try and do something sensible.
        if p0 is None:
            if verbose:
                logger.debug(f'Auto-initializing parameters for {self.name()}...')
            self.init_parameters(xdata, ydata)
            p0 = self.status.par_values
            if verbose:
                logger.debug(f'{self.name()} parameters initialized to {p0}.')
        # The actual call to the glorious scipy.optimize.curve_fit() method.
        popt, pcov = curve_fit(self, xdata, ydata, p0, sigma, absolute_sigma,
            check_finite, self.status.par_bounds, method, jac, **kwargs)
        # Update the model parameters.
        self.set_range(xdata.min(), xdata.max())
        chisq = (((ydata - self(xdata, *popt)) / sigma)**2).sum()
        ndof = len(ydata) - self.status.num_params
        self.status.update(popt, pcov, chisq, ndof)
        if verbose:
            print(self)

    def fit_histogram(self, histogram: Histogram1d, p0: np.ndarray=None,
        xmin: float = -np.inf, xmax: float = np.inf, absolute_sigma: bool = True,
        check_finite: bool = True, method: str = None, verbose: bool = True, **kwargs):
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
        """
        # pylint: disable=too-many-arguments
        if histogram.num_axes != 1:
            raise RuntimeError('Histogram is not one-dimensional')
        mask = (histogram.content > 0)
        xdata = histogram.bin_centers(0)[mask]
        ydata = histogram.content[mask]
        sigma = histogram.errors()[mask]
        return self.fit(xdata, ydata, p0, sigma, xmin, xmax, absolute_sigma, check_finite,
            method, verbose, **kwargs)

    @staticmethod
    def _merge_class_attributes(func, *components: type) -> Tuple:
        """Basic function to merge class attributes while summing models.

        This is heavily used in the model sum factory below, as it turns out that
        this is the besic signature that is needed to merge the class attributes
        when summing models.

        Note that we are not using the native Python ``sum(*args, start=[])``, here,
        as it is not supported in Python 3.7.
        """
        attrs = []
        for i, comp in enumerate(components):
            attrs += func(i, comp)
        return tuple(attrs)

    @staticmethod
    def model_sum_factory(*components: type) -> type:
        """Class factory to sum class models.

        Here we have worked out the math to sum up an arbitrary number of model
        classes.
        """
        mrg = FitModelBase._merge_class_attributes
        par_names = mrg(lambda i, c: [f'{name}{i}' for name in c.PAR_NAMES], *components)
        par_default_values = mrg(lambda i, c: list(c.PAR_DEFAULT_VALUES), *components)
        xmin = min([c.DEFAULT_RANGE[0] for c in components])
        xmax = max([c.DEFAULT_RANGE[1] for c in components])
        par_bound_min = mrg(lambda i, c: list(c.PAR_DEFAULT_BOUNDS[0]), *components)
        par_bound_max = mrg(lambda i, c: list(c.PAR_DEFAULT_BOUNDS[1]), *components)
        par_slices = []
        i = 0
        for c in components:
            num_params = len(c.PAR_NAMES)
            par_slices.append(slice(i, i + num_params))
            i += num_params

        class _model(FitModelBase):

            PAR_NAMES = par_names
            PAR_DEFAULT_VALUES = par_default_values
            DEFAULT_RANGE = (xmin, xmax)
            PAR_DEFAULT_BOUNDS = (par_bound_min, par_bound_max)

            def __init__(self):
                FitModelBase.__init__(self)

            @staticmethod
            def eval(x, *pars):
                """Overloaded method.
                """
                return sum([c.eval(x, *pars[s]) for c, s in zip(components, par_slices)])

            @staticmethod
            def jacobian(x, *pars):
                """Overloaded method.
                """
                return np.hstack([c.jacobian(x, *pars[s]) for c, s in zip(components, par_slices)])

            def plot(self, num_points: int = 250, **kwargs) -> None:
                """Overloaded method.

                In addition to the total model, here we overplot the single components.
                """
                x = np.linspace(self._xmin, self._xmax, num_points)
                plt.plot(x, self(x), **kwargs)
                color = last_line_color()
                for c, s in zip(components, par_slices):
                    plt.plot(x, c.eval(x, *self.status.par_values[s]),
                        ls='dashed', color=color)

        return _model

    def __add__(self, other):
        """Add two models.
        """
        model = self.model_sum_factory(self.__class__, other.__class__)
        model.__name__ = f'{self.name()} + {other.name()}'
        return model()

    def __str__(self):
        """String formatting.
        """
        return f'{self.name()} fit status\n{self.status}'



class Constant(FitModelBase):

    """Constant model.

    .. math::
       f(x; C) = C
    """

    PAR_NAMES = ('constant',)
    PAR_DEFAULT_VALUES = (1.,)

    @staticmethod
    def eval(x: np.ndarray, constant: float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return np.full(x.shape, constant)

    @staticmethod
    def jacobian(x: np.ndarray, constant: float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ, unused-argument
        d_constant = np.full((len(x),), 1.)
        return np.array([d_constant]).transpose()

    def init_parameters(self, xdata: np.ndarray, ydata: np.ndarray) -> None:
        """Overloaded method.
        """
        self.set_parameter('constant', np.mean(ydata))



class Line(FitModelBase):

    """Straight-line model.

    .. math::
       f(x; m, q) = mx + q
    """

    PAR_NAMES = ('intercept', 'slope')
    PAR_DEFAULT_VALUES = (1., 1.)

    @staticmethod
    def eval(x: np.ndarray, intercept: float, slope: float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return intercept + slope * x

    @staticmethod
    def jacobian(x: np.ndarray, intercept: float, slope: float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ, unused-argument
        d_intercept = np.full((len(x),), 1.)
        d_slope = x
        return np.array([d_intercept, d_slope]).transpose()



class Gaussian(FitModelBase):

    """One-dimensional Gaussian model.

    .. math::
      f(x; N, \\mu, \\sigma) = N e^{-\\frac{(x - \\mu)^2}{2\\sigma^2}}
    """

    PAR_NAMES = ('normalization', 'mean', 'sigma')
    PAR_DEFAULT_VALUES = (1., 0., 1.)
    PAR_DEFAULT_BOUNDS = ((0., -np.inf, 0.), (np.inf, np.inf, np.inf))
    DEFAULT_RANGE = (-5., 5.)
    SIGMA_TO_FWHM = 2.3548200450309493

    @staticmethod
    def eval(x: np.ndarray, normalization: float, mean: float, sigma: float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return normalization * np.exp(-0.5 * ((x - mean)**2. / sigma**2.))

    @staticmethod
    def jacobian(x: np.ndarray, normalization: float, mean: float, sigma: float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        d_normalization = np.exp(-0.5 / sigma**2. * (x - mean)**2.)
        d_mean = normalization * d_normalization * (x - mean) / sigma**2.
        d_sigma = normalization * d_normalization * (x - mean)**2. / sigma**3.
        return np.array([d_normalization, d_mean, d_sigma]).transpose()

    def init_parameters(self, xdata: np.ndarray, ydata: np.ndarray) -> None:
        """Overloaded method.
        """
        mean = np.average(xdata, weights=ydata)
        sigma = np.sqrt(np.average((xdata - mean)**2., weights=ydata))
        self.set_parameter('normalization', np.max(ydata))
        self.set_parameter('mean', mean)
        self.set_parameter('sigma', sigma)

    def fwhm(self) -> float:
        """Return the absolute FWHM of the model.
        """
        return self.SIGMA_TO_FWHM * self.status['sigma']



_DoubleGaussian = FitModelBase.model_sum_factory(Gaussian, Gaussian)

class DoubleGaussian(_DoubleGaussian):

    """Implementation of a double gaussian.
    """

    def init_parameters(self, xdata: np.ndarray, ydata: np.ndarray) -> None:
        """Overloaded method.
        """
        # Take all the data and make a first pass to a single-gaussian fit...
        model = Gaussian()
        # ... with a normalization and mean that are set to the position and value
        # of the highest input data point...
        norm = ydata.max()
        max_index = np.argmax(ydata)
        mean = xdata[max_index]
        # ... and the sigma is initialized moving away from the main peak
        # and gauging the FWHM
        y = norm
        i = 1
        while y > 0.5 * norm:
            y = 0.5 * (ydata[max_index - i] + ydata[max_index + i])
            i += 1
        sigma = abs(xdata[max_index - i] - xdata[max_index + i]) / Gaussian.SIGMA_TO_FWHM
        p0 = (norm, mean, sigma)
        logger.debug(f'First single-gaussian fit on all data with p0={p0}...')
        model.fit(xdata, ydata, p0=p0)
        # Second pass, where we refine the single gaussian fit to the main peak
        # starting from the parameters at the previous pass and restricting
        # ourselves to +/- 2 sigma around the peak.
        # Note that, since we are not passing the errors to this method, even
        # at this stage the chisquare makes no sense.
        norm, mean, sigma = model.status.par_values
        xmin = mean - 2. * sigma
        xmax = mean + 2. * sigma
        p0 = (norm, mean, sigma)
        logger.debug(f'Second single-gaussian fit in [{xmin}--{xmax}] with p0={p0}...')
        model.fit(xdata, ydata, p0=p0, xmin=xmin, xmax=xmax)
        # Cache the single-gaussian fit parameters at the second step, as they
        # are the initial parameters for the first gaussian in the model.
        self.set_parameter('normalization0', model['normalization'])
        self.set_parameter('mean0', model['mean'])
        self.set_parameter('sigma0', model['sigma'])
        # Now subtract the fitted main peak from the data and fit what
        # remains to another gaussian.
        y = ydata - model(xdata)
        # Set to zero all the stuff under the main peak, so that we don't risk
        # ending up on the fluctuations of the residuals.
        mask = np.logical_and(xdata > xmin, xdata < xmax)
        y[mask] = 0.
        logger.debug(f'Single gaussian fit on the secondary peak...')
        model.fit(xdata, y, p0=(y.max(), xdata[np.argmax(y)], sigma))
        self.set_parameter('normalization1', model['normalization'])
        self.set_parameter('mean1', model['mean'])
        self.set_parameter('sigma1', model['sigma'])

    def fit(self, *args, **kwargs):
        """Overloaded fit method.

        This is to ensure that, at the end of the fit, the order of the two gaussians
        is fixed, i.e., the one with the smallest mean come first.
        """
        super().fit(*args, **kwargs)
        if self.status.parameter_value('mean0') > self.status.parameter_value('mean1'):
            logger.debug('Swapping parameter sets...')
            # Note that we really need to make a full copy of the parameter
            # vector and the associated covariance matrix for the thing to work.
            popt = self.status.par_values.copy()
            pcov = self.status.par_covariance.copy()
            # Swap the parameter values.
            self.status.par_values[0:3] = popt[3:6]
            self.status.par_values[3:6] = popt[0:3]
            # Swap the proper block in the covariance matrix.
            self.status.par_covariance[0:3, 0:3] = pcov[3:6, 3:6]
            self.status.par_covariance[3:6, 3:6] = pcov[0:3, 0:3]
            self.status.par_covariance[0:3, 3:6] = pcov[3:6, 0:3]
            self.status.par_covariance[3:6, 0:3] = pcov[0:3, 3:6]



class PowerLaw(FitModelBase):

    """Power law model.

    .. math::
      f(x; N, \\Gamma) = N x^\\Gamma
    """

    PAR_NAMES = ('normalization', 'index')
    PAR_DEFAULT_VALUES = (1., -1.)
    PAR_DEFAULT_BOUNDS = ((0., -np.inf), (np.inf, np.inf))
    DEFAULT_RANGE = (1.e-2, 1.)

    @staticmethod
    def eval(x: np.ndarray, normalization: float, index: float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return normalization * x**index

    @staticmethod
    def jacobian(x: np.ndarray, normalization: float, index: float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        d_normalization = x**index
        d_index = np.log(x) * d_normalization * normalization
        return np.array([d_normalization, d_index]).transpose()



class Exponential(FitModelBase):

    """Exponential model.

    .. math::
      f(x; N, \\lambda) = N e^{\\frac{x}{\\lambda}}
    """

    PAR_NAMES = ('normalization', 'scale')
    PAR_DEFAULT_VALUES = (1., -1.)
    PAR_DEFAULT_BOUNDS = ((0., -np.inf), (np.inf, np.inf))

    @staticmethod
    def eval(x: np.ndarray, normalization: float, scale: float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return normalization * np.exp(-x / scale)

    @staticmethod
    def jacobian(x: np.ndarray, normalization: float, scale: float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        d_normalization = np.exp(-x / scale)
        d_scale = normalization * d_normalization * x / scale**2.
        return np.array([d_normalization, d_scale]).transpose()

    def init_parameters(self, xdata: np.ndarray, ydata: np.ndarray) -> None:
        """Overloaded method.
        """
        self.set_parameter('normalization', np.max(ydata))
        self.set_parameter('scale', np.average(xdata, weights=ydata))
