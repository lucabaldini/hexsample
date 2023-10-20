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
    PARAMETER_DEFAULT_BOUNDS = (-np.inf, np.inf)
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
        self.bounds = self.PARAMETER_DEFAULT_BOUNDS
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

    def stat_box(self, **kwargs) -> None:
        """Plot a stat box for the model.

        .. warning::
           This needs to be streamlined.
        """
        box = PlotCard()
        box.add_string('Fit model', self.name())
        box.add_string('Chisquare', f'{self.chisq:.1f} / {self.ndof}')
        for name, value, error in self:
            box.add_quantity(name, value, error)
        box.plot(**kwargs)

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

    #def integral(self, edges : np.ndarray) -> np.ndarray:
    #    """Calculate the integral of the model within pre-defined edges.
    #
    #    Note that this assumes that the derived class overloads the ``cdf()`` method.
    #    """
    #    return self.cdf(edges[1:]) - self.cdf(edges[:-1])

    @staticmethod
    def eval(x : np.ndarray, *parameters) -> np.ndarray:
        """Eval the model at a given x and a given set of parameter values.

        This needs to be overloaded by any derived classes.
        """
        raise NotImplementedError

    # def cumulative_function(self, x : np.ndarray) -> np.ndarray:
    #     """Return the cdf at a given x and a given set of parameter values.
    #
    #     This needs to be overloaded by any derived classes.
    #     """
    #     raise NotImplementedError

    def __add__(self, other):
        """Add two models.

        Warning
        -------
        This is highly experimental and guaranteed to contain bugs. Enjoy.
        """
        m1 = self
        m2 = other
        xmin = min(m1.DEFAULT_RANGE[0], m2.DEFAULT_RANGE[0])
        xmax = max(m1.DEFAULT_RANGE[1], m2.DEFAULT_RANGE[1])
        name = f'{m1.name()} + {m2.name()}'

        class _model(FitModelBase):

            PARAMETER_NAMES = [f'{name}1' for name in m1.PARAMETER_NAMES] + \
                [f'{name}2' for name in m2.PARAMETER_NAMES]
            PARAMETER_DEFAULT_VALUES = m1.PARAMETER_DEFAULT_VALUES + \
                m2.PARAMETER_DEFAULT_VALUES
            DEFAULT_RANGE = (xmin, xmax)
            PARAMETER_DEFAULT_BOUNDS = (-np.inf, np.inf)

            def __init__(self):
                self.__class__.__name__ = name
                FitModelBase.__init__(self)

            @staticmethod
            def eval(x, *parameters):
                return m1.eval(x, *parameters[:len(m1)]) +\
                    m2.eval(x, *parameters[len(m1):])

            @staticmethod
            def jacobian(x, *parameters):
                return np.hstack((m1.jacobian(x, *parameters[:len(m1)]),
                                     m2.jacobian(x, *parameters[len(m1):])))

        return _model()

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

    def cumulative_function(self, x : np.ndarray) -> np.ndarray:
        """Overloaded method.
        """
        return self['Constant'] * x

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
      f(x; A, \\mu, \\sigma) = A e^{-\\frac{(x - \\mu)^2}{2\\sigma^2}}
    """

    PARAMETER_NAMES = ('amplitude', 'mean', 'sigma')
    PARAMETER_DEFAULT_VALUES = (1., 0., 1.)
    PARAMETER_DEFAULT_BOUNDS = ((0., -np.inf, 0), (np.inf, np.inf, np.inf))
    DEFAULT_RANGE = (-5., 5.)
    SIGMA_TO_FWHM = 2.3548200450309493

    @staticmethod
    def eval(x : np.ndarray, amplitude : float, mean : float, sigma : float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return amplitude * np.exp(-0.5 * ((x - mean)**2. / sigma**2.))

    @staticmethod
    def jacobian(x : np.ndarray, amplitude : float, mean : float, sigma : float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        d_amplitude = np.exp(-0.5 / sigma**2. * (x - mean)**2.)
        d_mean = amplitude * d_amplitude * (x - mean) / sigma**2.
        d_sigma = amplitude * d_amplitude * (x - mean)**2. / sigma**3.
        return np.array([d_amplitude, d_mean, d_sigma]).transpose()

    def init_parameters(self, xdata : np.ndarray, ydata : np.ndarray, sigma : np.ndarray) -> None:
        """Overloaded method.
        """
        self.set_parameter('amplitude', np.max(ydata))
        self.set_parameter('mean', np.mean(xdata))
        self.set_parameter('sigma', np.std(xdata))

    def fwhm(self) -> float:
        """Return the absolute FWHM of the model.
        """
        return self.SIGMA_TO_FWHM * self['sigma']



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
        return normalization * (x**index)

    @staticmethod
    def jacobian(x : np.ndarray, normalization : float, index : float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        d_normalization = (x**index)
        d_index = np.log(x) * normalization * (x**index)
        return np.array([d_normalization, d_index]).transpose()



class Exponential(FitModelBase):

    """Exponential model.

    .. math::
      f(x; N, \\alpha) = N e^{\\alpha x}
    """

    PARAMETER_NAMES = ('normalization', 'scale')
    PARAMETER_DEFAULT_VALUES = (1., -1.)
    PARAMETER_DEFAULT_BOUNDS = ((0., -np.inf), (np.inf, np.inf))

    @staticmethod
    def eval(x : np.ndarray, normalization : float, scale : float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return normalization * np.exp(scale * x)

    @staticmethod
    def jacobian(x : np.ndarray, normalization : float, scale : float) -> np.ndarray:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        d_normalization = np.exp(scale * x)
        d_scale = normalization * x * np.exp(scale * x)
        return np.array([d_normalization, d_scale]).transpose()
