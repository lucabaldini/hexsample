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

from hexsample.plot import plt, PlotCard

# pylint: disable=invalid-name


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
    ``init_parameters(xdata, ydata, sigma)`` method of the base class, as the
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
        # Make sure the length of the parameter names and values match.
        if len(self.PARAMETER_NAMES) != len(self.PARAMETER_DEFAULT_VALUES):
            raise RuntimeError(f'Mismatch between parameter names and values for {self.name()}')
        # Initialize the basic class member---the parameters are set to their default values...
        self.parameters = np.array(self.PARAMETER_DEFAULT_VALUES, dtype='d')
        self.num_parameters = len(self.parameters)
        # ... the covariance matrix is all zeros...
        self.covariance_matrix = np.zeros((self.num_parameters, self.num_parameters), dtype='d')
        # ... the bounds are set to their defalut values...
        self.bounds = np.array(self.PARAMETER_DEFAULT_BOUNDS)
        # .. and so it's the range...
        self._xmin, self._xmax = self.DEFAULT_RANGE
        # ... and the chisquare and number of degrees of freedom are set to -1.
        self.chisq = -1.
        self.ndof = -1
        # Create a small lookup table between parameter names and their position
        # within the array of parameter values.
        self._param_index_dict = {name : i for i, name in enumerate(self.PARAMETER_NAMES)}

    def name(self) -> str:
        """Return the model name.
        """
        return self.__class__.__name__

    def _parameter_index(self, parameter_name : str) -> int:
        """Convenience method returning the index within the parameter vector
        for a given parameter name.
        """
        try:
            return self._param_index_dict[parameter_name]
        except KeyError as exception:
            raise KeyError(f'Unknown parameter "{parameter_name}" for {self.name()}') from exception

    def parameter_values(self) -> np.ndarray:
        """Return the vector of parameter values---this is just a convenience function
        provided for consistency with the following ``parameter_error()``.
        """
        return self.parameters

    def parameter_errors(self) -> np.ndarray:
        """Return the vector of parameter errors, that is, the square root of the
        diagonal elements of the covariance matrix.
        """
        return np.sqrt(self.covariance_matrix.diagonal())

    def __iter__(self):
        """Iterate over the underlying parameters as (name, value, error).

        This comes handy, e.g., for printing the best fit values on the terminal,
        or for creating a stat box.
        """
        return zip(self.PARAMETER_NAMES, self.parameter_values(), self.parameter_errors())

    def parameter_value(self, parameter_name : str) -> float:
        """Return the parameter value by name.
        """
        return self.parameters[self._parameter_index(parameter_name)]

    def __getitem__(self, parameter_name : str) -> float:
        """Convenience shortcut to retrieve the value of a parameter.
        """
        return self.parameter_value(parameter_name)

    def parameter_error(self, parameter_name : str) -> float:
        """Return the parameter error by name.
        """
        parameter_index = self._parameter_index(parameter_name)
        return np.sqrt(self.covariance_matrix[parameter_index][parameter_index])

    def set_parameter(self, parameter_name : str, value : float) -> None:
        """Set the value for a given parameter (indexed by name).
        """
        self.parameters[self._parameter_index(parameter_name)] = value

    def init_parameters(self, xdata : np.ndarray, ydata : np.ndarray, sigma : np.ndarray) -> None:
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

        sigma : array_like
            The uncertainties on the y data.
        """

    def reduced_chisquare(self) -> float:
        """Return the reduced chisquare.
        """
        if self.ndof > 0:
            return self.chisq / self.ndof
        return -1.

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
        box.add_string('Chisquare', f'{self.chisq:.1f} / {self.ndof}')
        for name, value, error in self:
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
        if num_params == self.num_parameters:
            return self.eval(x, *parameters)
        if num_params == 0:
            return self.eval(x, *self.parameters)
        raise RuntimeError(f'Wrong number of parameters ({num_params}) to {self.name()}.__call__()')

    @staticmethod
    def eval(x : np.ndarray, *parameters) -> np.ndarray:
        """Eval the model at a given x and a given set of parameter values.

        This needs to be overloaded by any derived classes.
        """
        raise NotImplementedError

    @staticmethod
    def _merge_class_attributes(func, *models : type) -> Tuple:
        """Basic function to merge class attributes while summing models.

        This is heavily used in the model sum factory below, as it turns out that
        this is the besic signature that is needed to merge the class attributes
        when summing models.
        """
        return tuple(sum([func(i, model) for i, model in enumerate(models)], start=[]))

    @staticmethod
    def model_sum_factory(*models : type) -> type:
        """Class factory to sum class models.

        Here we have worked out the math to sum up an arbitrary number of model
        classes.
        """
        name = ' + '.join([model.__name__ for model in models])
        mrg = FitModelBase._merge_class_attributes
        par_names = mrg(lambda i, m: [f'{name}{i}' for name in m.PARAMETER_NAMES], *models)
        par_default_values = mrg(lambda i, m: list(m.PARAMETER_DEFAULT_VALUES), *models)
        xmin = min([m.DEFAULT_RANGE[0] for m in models])
        xmax = max([m.DEFAULT_RANGE[1] for m in models])
        par_bound_min = mrg(lambda i, m: list(m.PARAMETER_DEFAULT_BOUNDS[0]), *models)
        par_bound_max = mrg(lambda i, m: list(m.PARAMETER_DEFAULT_BOUNDS[1]), *models)
        par_slices = []
        i = 0
        for m in models:
            num_parameters = len(m.PARAMETER_NAMES)
            par_slices.append(slice(i, i + num_parameters))
            i += num_parameters

        class _model(FitModelBase):

            PARAMETER_NAMES = par_names
            PARAMETER_DEFAULT_VALUES = par_default_values
            DEFAULT_RANGE = (xmin, xmax)
            PARAMETER_DEFAULT_BOUNDS = (par_bound_min, par_bound_max)

            def __init__(self):
                self.__class__.__name__ = name
                FitModelBase.__init__(self)

            @staticmethod
            def eval(x, *pars):
                return sum([m.eval(x, *pars[s]) for m, s in zip(models, par_slices)])

            @staticmethod
            def jacobian(x, *pars):
                return np.hstack([m.jacobian(x, *pars[s]) for m, s in zip(models, par_slices)])

        return _model

    def __add__(self, other):
        """Add two models.
        """
        return self.model_sum_factory(self.__class__, other.__class__)()

    def __str__(self):
        """String formatting.
        """
        text = f'{self.name()} (chisq/ndof = {self.chisq:.2f} / {self.ndof})'
        for name, value, error in self:
            text += f'\n{name:15s}: {value:.5e} +- {error:.5e}'
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

    def init_parameters(self, xdata : np.ndarray, ydata : np.ndarray, sigma : np.ndarray) -> None:
        """Overloaded method.
        """
        self.set_parameter('constant', np.mean(ydata))



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

    def init_parameters(self, xdata : np.ndarray, ydata : np.ndarray, sigma : np.ndarray) -> None:
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
        return self.SIGMA_TO_FWHM * self['sigma']



_DoubleGaussian = FitModelBase.model_sum_factory(Gaussian, Gaussian)

class DoubleGaussian(_DoubleGaussian):

    """
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

    def init_parameters(self, xdata : np.ndarray, ydata : np.ndarray, sigma : np.ndarray) -> None:
        """Overloaded method.
        """
        self.set_parameter('normalization', np.max(ydata))
        self.set_parameter('scale', np.average(xdata, weights=ydata))



if __name__ == '__main__':
    #m = FitModelBase.model_sum_factory(Constant, Gaussian, Gaussian)
    m = DoubleGaussian()
    m.parameters = np.array([1., 10., 1., 1., 15., 1.])
    m.set_range(5., 20.)
    m.plot()
    m.stat_box()
    plt.show()
