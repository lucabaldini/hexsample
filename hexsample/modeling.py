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
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, parameter_names : Tuple, parameter_values : np.ndarray,
        parameter_bounds : Tuple = None) -> None:
        """Constructor.
        """
        if len(parameter_names) != len(parameter_values):
            raise RuntimeError('Mismatch between parameter names and values')
        self.parameter_names = parameter_names
        self.parameter_values = np.array(parameter_values)
        self.num_parameters = len(self.parameter_names)
        self.parameter_bounds = self._process_bounds(parameter_bounds)
        self.covariance_matrix = np.zeros((self.num_parameters, self.num_parameters), dtype=float)
        self._parameter_free = np.ones(self.num_parameters, dtype=bool)
        self.chisquare = -1.
        self.ndof = -1
        self._parameter_index_dict = {name : i for i, name in enumerate(self.parameter_names)}

    def _process_bounds(self, parameter_bounds : Tuple = None) -> Tuple[np.ndarray, np.ndarray]:
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
        if parameter_bounds is None:
            return (np.full(self.num_parameters, -np.inf), np.full(self.num_parameters, np.inf))
        if len(parameter_bounds) != 2:
            raise RuntimeError(f'Invalid parameter bounds {parameter_bounds}')
        return tuple([np.array(bounds) for bounds in parameter_bounds])

    def _parameter_index(self, parameter_name : str) -> int:
        """Convenience method returning the index within the parameter vector
        for a given parameter name.
        """
        try:
            return self._parameter_index_dict[parameter_name]
        except KeyError as exception:
            raise KeyError(f'Unknown parameter "{parameter_name}"') from exception

    def parameter_value(self, parameter_name : str) -> float:
        """Return the parameter value for a given parameter indexed by name.
        """
        return self.parameter_values[self._parameter_index(parameter_name)]

    def __getitem__(self, parameter_name : str) -> float:
        """Convenience shortcut to retrieve the value of a parameter.
        """
        return self.parameter_value(parameter_name)

    def parameter_errors(self) -> np.ndarray:
        """Return the vector of parameter errors, that is, the square root of the
        diagonal elements of the covariance matrix.
        """
        return np.sqrt(self.covariance_matrix.diagonal())

    def parameter_error(self, parameter_name : str) -> float:
        """Return the parameter error by name.
        """
        index = self._parameter_index(parameter_name)
        return np.sqrt(self.covariance_matrix[index][index])

    # def parameter_free(self, parameter_name : str) -> float:
    #     """Return True if the given parameter is free to vary in the fit.
    #     """
    #     return self._parameter_free[self._parameter_index(parameter_name)]

    def reduced_chisquare(self) -> float:
        """Return the reduced chisquare.
        """
        return self.chisquare / self.ndof if self.ndof > 0 else -1.

    def set_parameter_bounds(self, parameter_name : str, min_bound : float,
        max_bound : float) -> None:
        """Set the baounds for a given parameter.
        """
        index = self._parameter_index(parameter_name)
        self.parameter_bounds[0][index] = min_bound
        self.parameter_bounds[1][index] = max_bound

    def set_parameter(self, parameter_name : str, value : float) -> None:
        """Set the value for a given parameter (indexed by name).
        """
        self.parameter_values[self._parameter_index(parameter_name)] = value

    def update_parameter(self, parameter_name : str, value : float) -> None:
        """Alias, waiting for the implementation of frozen parameters.
        """
        self.set_parameter(parameter_name, value)

    # def update_parameter(self, parameter_name : str, value : float) -> None:
    #     """Update a parameter value.
    #
    #     This is calling the set_parameter() hook *only* if the the parameter
    #     itself is free to vary, and should be the default choice to interact
    #     with the parameter values in the implementation of FitModelBase.init_parameters()
    #     by concrete subclasses.
    #     """
    #     if self.parameter_free(parameter_name):
    #         self.set_parameter(parameter_name, value)
    #
    # def freeze_parameter(self, parameter_name : str, value : float) -> None:
    #     """Freeze one of the parameters at a given value.
    #     """
    #     index = self._parameter_index(parameter_name)
    #     self.parameter_values[index] = value
    #     self._parameter_free[index] = False

    def __iter__(self):
        """Iterate over the fit parameters as (name, value, error).
        """
        return zip(self.parameter_names, self.parameter_values, self.parameter_errors(),\
            *self.parameter_bounds, self._parameter_free)

    def __str__(self):
        """String representation.
        """
        text = ''
        for name, value, error, min_bound, max_bound, free in self:
            text += f'{name:18s} {value:.3e} +/- {error:.3e} ({min_bound:.3e} / {max_bound:.3e})'
            if not free:
                text += ' fixed'
            text += '\n'
        text += f'Chisquare = {self.chisquare} / {self.ndof} dof'
        return text



class FitModelBase:

    """Base class for a fittable model.

    The class features a number of static members that derived class should redefine
    as needed:

    * ``PARAMETER_NAMES`` is a list containing the names of the model parameters.
    * ``PARAMETER_DEFAULT_VALUES`` is a list containing the default values
      of the model parameters---when a concrete model object is instantiated
      these are the values being attached to it at creation time.
    * ``PARAMETER_DEFAULT_BOUNDS`` is a tuple containing the default values
      of the parameter bounds to be used for the fitting. The values in the
      tuple are attached to each model object at creation time and are
      intended to be passed as the ``bounds`` argument of the
      ``scipy.optimize.curve_fit()`` function. From the ``scipy`` documentation:
      Lower and upper bounds on independent variables. Defaults to no bounds.
      Each element of the tuple must be either an array with the length equal
      to the number of parameters, or a scalar (in which case the bound is
      taken to be the same for all parameters.) Use np.inf with an appropriate
      sign to disable bounds on all or some parameters. By default
      models have no built-in bounds.
    * ``DEFAULT_RANGE`` is a two-element list with the default support
      (x-axis range) for the model. This is automatically updated at runtime
      depending on the input data when the model is used in a fit.

   In addition, each derived class should override the following methods:

    * the ``eval(x, *args)`` static method: this should return the value of
      the model at a given point for a given set of values of the underlying
      parameters;
    * (optionally) the ``jacobian(x, *args)`` static method. (If defined, this
      is passed to the underlying fit engine allowing to reduce the number of
      function calls in the fit; otherwise the jacobian is calculated
      numerically.)

    Finally, if there is a sensible way to initialize the model parameters
    based on a set of input data, derived classes should overload the
    ``init_parameters(xdata, ydata)`` method of the base class, as the
    latter is called by fitting routines if no explicit array of initial values
    are passed as an argument. The default behavior of the class method defined
    in the base class is to do nothing.

    See :class:`hexsample.modeling.Gaussian` for a working example.
    """

    # pylint: disable=too-many-instance-attributes

    PARAMETER_NAMES = None
    PARAMETER_DEFAULT_VALUES = None
    PARAMETER_DEFAULT_BOUNDS = None
    DEFAULT_RANGE = (0., 1.)

    def __init__(self) -> None:
        """Constructor.
        """
        args = self.PARAMETER_NAMES, self.PARAMETER_DEFAULT_VALUES, self.PARAMETER_DEFAULT_BOUNDS
        self.status = FitStatus(*args)
        self._xmin, self._xmax = self.DEFAULT_RANGE

    def name(self) -> str:
        """Return the model name.
        """
        return self.__class__.__name__

    def init_parameters(self, xdata : np.ndarray, ydata : np.ndarray) -> None:
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

    def set_range(self, xmin : float, xmax : float) -> None:
        """Set the function range.
        """
        self._xmin = xmin
        self._xmax = xmax

    def plot(self, num_points : int = 250, **kwargs) -> None:
        """Plot the model.
        """
        x = np.linspace(self._xmin, self._xmax, num_points)
        plt.plot(x, self(x), **kwargs)

    def stat_box(self, x : float = 0.95, y : float = 0.95) -> None:
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
        if num_params == self.status.num_parameters:
            return self.eval(x, *parameters)
        if num_params == 0:
            return self.eval(x, *self.status.parameter_values)
        raise RuntimeError(f'Wrong number of parameters ({num_params}) to {self.name()}.__call__()')

    @staticmethod
    def eval(x : np.ndarray, *parameters) -> np.ndarray:
        """Eval the model at a given x and a given set of parameter values.

        This needs to be overloaded by any derived classes.
        """
        raise NotImplementedError

    def _calculate_chisquare(self, xdata : np.ndarray, ydata : np.ndarray, sigma : np.ndarray) -> float:
        """Calculate the chisquare for the current parameter values, given
        some input data.
        """
        return (((ydata - self(xdata)) / sigma)**2).sum()

    def fit(self, xdata : np.ndarray, ydata : np.ndarray, p0 : np.ndarray = None,
        sigma : np.ndarray = None, xmin : float = -np.inf, xmax : float = np.inf,
        absolute_sigma : bool = True, check_finite : bool =True, method : str = None,
        verbose : bool = True, **kwargs):
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
        mask = np.logical_and(xdata >= xmin, xdata <= xmax)
        xdata = xdata[mask]
        if len(xdata) <= self.status.num_parameters:
            raise RuntimeError(f'Not enough data to fit ({len(xdata)} points)')
        ydata = ydata[mask]
        # If sigma is None, assume all the errors are 1---we need to do this
        # explicitely in order to calculate the chisquare downstream.
        if sigma is None:
            sigma = np.full(len(ydata), 1.)
        sigma = sigma[mask]
        # If the model has a Jacobian defined, go ahead and use it.
        try:
            jac = self.jacobian
        except:
            jac = None
        # If we are not passing default starting points for the model parameters,
        # try and do something sensible.
        if p0 is None:
            self.init_parameters(xdata, ydata)
            p0 = self.status.parameter_values
            if verbose:
                logger.debug(f'{self.name()} parameters initialized to {p0}.')
        # The actual call to the glorious scipy.optimize.curve_fit() method.
        popt, pcov = curve_fit(self, xdata, ydata, p0, sigma, absolute_sigma,
            check_finite, self.status.parameter_bounds, method, jac, **kwargs)
        # Update the model parameters.
        self.set_range(xdata.min(), xdata.max())
        self.status.parameter_values = popt
        self.status.covariance_matrix = pcov
        self.status.chisquare = self._calculate_chisquare(xdata, ydata, sigma)
        self.status.ndof = len(ydata) - self.status.num_parameters
        if verbose:
            print(self)

    def fit_histogram(self, histogram : Histogram1d, p0 : np.ndarray=None,
        xmin : float = -np.inf, xmax : float = np.inf, absolute_sigma : bool = True,
        check_finite : bool = True, method : str = None, verbose : bool = True, **kwargs):
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
        if histogram.num_axes != 1:
            raise RuntimeError(f'Histogram is not one-dimensional')
        mask = (histogram.content > 0)
        xdata = histogram.bin_centers(0)[mask]
        ydata = histogram.content[mask]
        sigma = histogram.errors()[mask]
        return self.fit(xdata, ydata, p0, sigma, xmin, xmax, absolute_sigma, check_finite,
            method, verbose, **kwargs)

    @staticmethod
    def _merge_class_attributes(func, *components : type) -> Tuple:
        """Basic function to merge class attributes while summing models.

        This is heavily used in the model sum factory below, as it turns out that
        this is the besic signature that is needed to merge the class attributes
        when summing models.

        Note that we are not using the native Python sum(*args, start=[]), here,
        as it is not supported in Python 3.7.
        """
        attrs = []
        for i, comp in enumerate(components):
            attrs += func(i, comp)
        return tuple(attrs)

    @staticmethod
    def model_sum_factory(*components : type) -> type:
        """Class factory to sum class models.

        Here we have worked out the math to sum up an arbitrary number of model
        classes.
        """
        mrg = FitModelBase._merge_class_attributes
        par_names = mrg(lambda i, c: [f'{name}{i}' for name in c.PARAMETER_NAMES], *components)
        par_default_values = mrg(lambda i, c: list(c.PARAMETER_DEFAULT_VALUES), *components)
        xmin = min([c.DEFAULT_RANGE[0] for c in components])
        xmax = max([c.DEFAULT_RANGE[1] for c in components])
        par_bound_min = mrg(lambda i, c: list(c.PARAMETER_DEFAULT_BOUNDS[0]), *components)
        par_bound_max = mrg(lambda i, c: list(c.PARAMETER_DEFAULT_BOUNDS[1]), *components)
        par_slices = []
        i = 0
        for c in components:
            num_parameters = len(c.PARAMETER_NAMES)
            par_slices.append(slice(i, i + num_parameters))
            i += num_parameters

        class _model(FitModelBase):

            PARAMETER_NAMES = par_names
            PARAMETER_DEFAULT_VALUES = par_default_values
            DEFAULT_RANGE = (xmin, xmax)
            PARAMETER_DEFAULT_BOUNDS = (par_bound_min, par_bound_max)

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

            def plot(self, num_points : int = 250, **kwargs) -> None:
                """Overloaded method.

                In addition to the total model, here we overplot the single components.
                """
                x = np.linspace(self._xmin, self._xmax, num_points)
                plt.plot(x, self(x), **kwargs)
                color = last_line_color()
                for c, s in zip(components, par_slices):
                    plt.plot(x, c.eval(x, *self.status.parameter_values[s]),
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
        text = f'{self.name()} (chisq/ndof = {self.status.chisquare:.2f} / {self.status.ndof})\n{self.status}'
        return text



class Constant(FitModelBase):

    """Constant model.

    .. math::
       f(x; C) = C
    """

    PARAMETER_NAMES = ('constant',)
    PARAMETER_DEFAULT_VALUES = (1.,)
    PARAMETER_DEFAULT_BOUNDS = ((-np.inf,), (np.inf,))

    @staticmethod
    def eval(x : np.ndarray, constant : float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return np.full(x.shape, constant)

    @staticmethod
    def jacobian(x : np.ndarray, constant: float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ, unused-argument
        d_constant = np.full((len(x),), 1.)
        return np.array([d_constant]).transpose()

    def init_parameters(self, xdata : np.ndarray, ydata : np.ndarray) -> None:
        """Overloaded method.
        """
        self.status.update_parameter('constant', np.mean(ydata))



class Line(FitModelBase):

    """Straight-line model.

    .. math::
       f(x; m, q) = mx + q
    """

    PARAMETER_NAMES = ('intercept', 'slope')
    PARAMETER_DEFAULT_VALUES = (1., 1.)
    PARAMETER_DEFAULT_BOUNDS = ((-np.inf,), (np.inf,))

    @staticmethod
    def eval(x : np.ndarray, intercept : float, slope : float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return intercept + slope * x

    @staticmethod
    def jacobian(x : np.ndarray, intercept : float, slope : float) -> np.ndarray:
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

    PARAMETER_NAMES = ('normalization', 'mean', 'sigma')
    PARAMETER_DEFAULT_VALUES = (1., 0., 1.)
    PARAMETER_DEFAULT_BOUNDS = ((0., -np.inf, 0.), (np.inf, np.inf, np.inf))
    DEFAULT_RANGE = (-5., 5.)
    SIGMA_TO_FWHM = 2.3548200450309493

    @staticmethod
    def eval(x : np.ndarray, normalization : float, mean : float, sigma : float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return normalization * np.exp(-0.5 * ((x - mean)**2. / sigma**2.))

    @staticmethod
    def jacobian(x : np.ndarray, normalization : float, mean : float, sigma : float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        d_normalization = np.exp(-0.5 / sigma**2. * (x - mean)**2.)
        d_mean = normalization * d_normalization * (x - mean) / sigma**2.
        d_sigma = normalization * d_normalization * (x - mean)**2. / sigma**3.
        return np.array([d_normalization, d_mean, d_sigma]).transpose()

    def init_parameters(self, xdata : np.ndarray, ydata : np.ndarray) -> None:
        """Overloaded method.
        """
        mean = np.average(xdata, weights=ydata)
        sigma = np.sqrt(np.average((xdata - mean)**2., weights=ydata))
        self.status.update_parameter('normalization', np.max(ydata))
        self.status.update_parameter('mean', mean)
        self.status.update_parameter('sigma', sigma)

    def fwhm(self) -> float:
        """Return the absolute FWHM of the model.
        """
        return self.SIGMA_TO_FWHM * self.status['sigma']



_DoubleGaussian = FitModelBase.model_sum_factory(Gaussian, Gaussian)

class DoubleGaussian(_DoubleGaussian):

    """Implementation of a double gaussian.
    """



class PowerLaw(FitModelBase):

    """Power law model.

    .. math::
      f(x; N, \\Gamma) = N x^\\Gamma
    """

    PARAMETER_NAMES = ('normalization', 'index')
    PARAMETER_DEFAULT_VALUES = (1., -1.)
    PARAMETER_DEFAULT_BOUNDS = ((0., -np.inf), (np.inf, np.inf))
    DEFAULT_RANGE = (1.e-2, 1.)

    @staticmethod
    def eval(x : np.ndarray, normalization : float, index : float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return normalization * x**index

    @staticmethod
    def jacobian(x : np.ndarray, normalization : float, index : float) -> np.ndarray:
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

    PARAMETER_NAMES = ('normalization', 'scale')
    PARAMETER_DEFAULT_VALUES = (1., -1.)
    PARAMETER_DEFAULT_BOUNDS = ((0., -np.inf), (np.inf, np.inf))

    @staticmethod
    def eval(x : np.ndarray, normalization : float, scale : float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return normalization * np.exp(-x / scale)

    @staticmethod
    def jacobian(x : np.ndarray, normalization : float, scale : float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        d_normalization = np.exp(-x / scale)
        d_scale = normalization * d_normalization * x / scale**2.
        return np.array([d_normalization, d_scale]).transpose()

    def init_parameters(self, xdata : np.ndarray, ydata : np.ndarray) -> None:
        """Overloaded method.
        """
        self.status.update_parameter('normalization', np.max(ydata))
        self.status.update_parameter('scale', np.average(xdata, weights=ydata))
