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

from __future__ import print_function, division


import numpy as np

from hexsample.plot import plt, PlotCard

# pylint: disable=invalid-name


class FitModelBase:

    """Base class for a fittable model.

    This base class isn't really doing anything useful, the idea being that
    actual models that can be instantiated subclass FitModelBase overloading
    the relevant class members.

    The class features a number of static members that derived class
    should redefine as needed:

    * ``PARAMETER_NAMES`` is a list containing the names of the model
      parameters. (It goes without saying thet its length should match the
      number of parameters in the model.)
    * ``PARAMETER_DEFAULT_VALUES`` is a list containing the default values
      of the model parameters. When a concrete model object is instantiated
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
    * ``DEFAULT_PLOTTING_RANGE`` is a two-element list with the default support
      (x-axis range) for the model. This is automatically updated at runtime
      depending on the input data when the model is used in a fit.
    * ``DEFAULT_STAT_BOX_POSITION`` is the default location of the stat box for
      the model.

   In addition, each derived class should override the following things:

    * the ``value(x, *args)`` static method: this should return the value of
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

    See :class:`modeling.Gaussian` for a working example.
    """

    PARAMETER_NAMES = ()
    PARAMETER_DEFAULT_VALUES = ()
    PARAMETER_DEFAULT_BOUNDS = (-np.inf, np.inf)
    DEFAULT_PLOTTING_RANGE = (0., 1.)
    DEFAULT_STAT_BOX_POSITION = 'upper right'

    def __init__(self):
        """Constructor.

        Here we initialize the class members holding the best-fit parameter
        values and the associated covariance matrix, see the
        :pymeth:`modeling.FitModelBase.reset()` method.

        We also create a (private) look-up table holding the correspondence
        between the parameter names and the corresponding position in the
        parameter list and we cache the default support for the model for
        plotting purposes.
        """
        assert len(self.PARAMETER_NAMES) == len(self.PARAMETER_DEFAULT_VALUES)
        self.__parameter_dict = {}
        for i, name in enumerate(self.PARAMETER_NAMES):
            self.__parameter_dict[name] = i
        self.reset()

    def name(self):
        """Return the model name.
        """
        return self.__class__.__name__

    def reset(self):
        """Reset all the necessary stuff.

        This method initializes all the things that are necessry to keep
        track of a parametric fit.

        * the parameter values are set to what it specified in
          ``PARAMETER_DEFAULT_VALUES``;
        * the covatiance matrix is initialized to a matrix of the proper
          dimension filled with zeroes;
        * the minimum and maximum values of the independent variable
          (relevant for plotting) are set to the values specified in
          ``DEFAULT_PLOTTING_RANGE``;
        * the model bounds are set to the values specified in
          ``PARAMETER_DEFAULT_BOUNDS``;
        * the chisquare and number of degrees of freedom are initialized to
          -1.
        """
        self.parameters = np.array(self.PARAMETER_DEFAULT_VALUES, dtype='d')
        self.covariance_matrix = np.zeros((len(self), len(self)), dtype='d')
        self.xmin, self.xmax = self.DEFAULT_PLOTTING_RANGE
        self.bounds = self.PARAMETER_DEFAULT_BOUNDS
        self.chisq = -1.
        self.ndof = -1

    def reduced_chisquare(self):
        """Return the reduced chisquare.
        """
        if self.ndof > 0:
            return self.chisq / self.ndof
        return -1.

    def __getattr__(self, name):
        """Short-hand method to retrieve the parameter values by name.

        Note that we manipulate the attribute name by capitalizing the
        first letter and replacing underscores with spaces.
        """
        name = name.title().replace('_', ' ')
        if name in self.PARAMETER_NAMES:
            return self.parameter_value(name)
        else:
            raise AttributeError

    @staticmethod
    def value(x, *parameters):
        """Eval the model at a given point and a given set of parameter values.

        Warning
        -------
        This needs to be overloaded in any derived class for the thing to do
        something sensible.
        """
        raise 'value() not implemented'

    def integral(self, edges):
        """Calculate the integral of the model within pre-defined edges.

        Note that this assumes that the derived class provides a suitable
        ``cdf()`` method.
        """
        try:
            return self.cdf(edges[1:], *self.parameters) - \
                self.cdf(edges[:-1], *self.parameters)
        except Exception as e:
            raise RuntimeError('%s.integral() not implemened (%s)' % (self.name, e))

    def rvs(self, size=1):
        """Return random variates from the model.

        Note that this assumes that the derived class provides a suitable
        ``ppf()`` method.
        """
        try:
            return self.ppf(np.random.random(size), *self.parameters)
        except Exception as e:
            raise RuntimeError('%s.rvs() not implemened (%s)' % (self.name(), e))

    def __call__(self, x, *parameters):
        """Return the value of the model at a given point and a given set of
        parameter values.

        Note that unless the proper number of parameters is passed to the
        function call, the model is evaluated at the best-fit parameter values.

        The function is defined with this signature because it is called
        with a set of parameter values during the fit process, while
        tipically we want to evaluate it with the current set of parameter
        values after the fact.
        """
        if len(parameters) == len(self):
            return self.value(x, *parameters)
        else:
            return self.value(x, *self.parameters)

    def __parameter_index(self, name):
        """Convenience method returning the index within the parameter vector
        for a given parameter name.
        """
        assert(name in self.PARAMETER_NAMES)
        return self.__parameter_dict[name]

    def parameter_value(self, name):
        """Return the parameter value by name.
        """
        index = self.__parameter_index(name)
        return self.parameters[index]

    def parameter_error(self, name):
        """Return the parameter error by name.
        """
        index = self.__parameter_index(name)
        return np.sqrt(self.covariance_matrix[index][index])

    def parameter_values(self):
        """Return the vector of parameter values.
        """
        return self.parameters

    def parameter_errors(self):
        """Return the vector of parameter errors.
        """
        return np.sqrt(self.covariance_matrix.diagonal())

    def parameter_status(self):
        """Return the complete status of the model in the form of a tuple
        of tuples (parameter_name, parameter_value, parameter_error).

        Note this can be overloaded by derived classes if more information
        needs to be added.
        """
        return tuple(zip(self.PARAMETER_NAMES, self.parameter_values(),
                         self.parameter_errors()))

    def set_parameters(self, *pars):
        """Set all the parameter values.

        Note that the arguments must be passed in the right order.
        """
        self.parameters = np.array(pars, dtype='d')

    def set_parameter(self, name, value):
        """Set a parameter value.
        """
        index = self.__parameter_index(name)
        self.parameters[index] = value

    def init_parameters(self, xdata, ydata, sigma):
        """Assign a sensible set of values to the model parameters, based
        on a data set to be fitted.

        Note that in the base class the method is not doing anything, but it
        can be reimplemented in derived classes to help make sure the
        fit converges without too much manual intervention.
        """
        pass

    def set_plotting_range(self, xmin, xmax):
        """Set the plotting range.
        """
        self.xmin = xmin
        self.xmax = xmax

    def plot(self, *parameters, **kwargs):
        """Plot the model.

        Note that when this is called with a full set of parameters, the
        self.parameters class member is overwritten so that the right values
        can then be picked up if the stat box is plotted.
        """
        if len(parameters) == len(self):
            self.parameters = parameters
        x = np.linspace(self.xmin, self.xmax, 1000)
        y = self(x, *parameters)
        plt.plot(x, y, **kwargs)

    def stat_box(self, **kwargs):
        """Plot a ROOT-style stat box for the model.
        """
        box = PlotCard()
        box.add_string('Fit model', self.name())
        box.add_string('Chisquare', '%.1f / %d' % (self.chisq, self.ndof))
        for name, value, error in self.parameter_status():
            box.add_quantity(name, value, error)
        box.plot(**kwargs)

    def __len__(self):
        """Return the number of model parameters.
        """
        return len(self.PARAMETER_NAMES)

    def __add__(self, other):
        """Add two models.

        Warning
        -------
        This is highly experimental and guaranteed to contain bugs. Enjoy.
        """
        m1 = self
        m2 = other
        xmin = min(m1.DEFAULT_PLOTTING_RANGE[0], m2.DEFAULT_PLOTTING_RANGE[0])
        xmax = max(m1.DEFAULT_PLOTTING_RANGE[1], m2.DEFAULT_PLOTTING_RANGE[1])
        name = '%s + %s' % (m1.__class__.__name__, m2.__class__.__name__)

        class _model(FitModelBase):

            PARAMETER_NAMES = [f'{name}1' for name in m1.PARAMETER_NAMES] + \
                [f'{name}2' for name in m2.PARAMETER_NAMES]
            PARAMETER_DEFAULT_VALUES = m1.PARAMETER_DEFAULT_VALUES + \
                m2.PARAMETER_DEFAULT_VALUES
            DEFAULT_PLOTTING_RANGE = (xmin, xmax)
            PARAMETER_DEFAULT_BOUNDS = (-np.inf, np.inf)

            def __init__(self):
                self.__class__.__name__ = name
                FitModelBase.__init__(self)

            @staticmethod
            def value(x, *parameters):
                return m1.value(x, *parameters[:len(m1)]) +\
                    m2.value(x, *parameters[len(m1):])

            @staticmethod
            def jacobian(x, *parameters):
                return np.hstack((m1.jacobian(x, *parameters[:len(m1)]),
                                     m2.jacobian(x, *parameters[len(m1):])))

        return _model()

    def __str__(self):
        """String formatting.
        """
        text = '%s model (chisq/ndof = %.2f / %d)' % (self.__class__.__name__,
                                                      self.chisq, self.ndof)
        for name, value, error in self.parameter_status():
            text += '\n%15s: %.5e +- %.5e' % (name, value, error)
        return text


class Constant(FitModelBase):

    """Constant model.

    .. math::
      f(x; C) = C
    """

    PARAMETER_NAMES = ('Constant',)
    PARAMETER_DEFAULT_VALUES = (1.,)
    DEFAULT_PLOTTING_RANGE = (0., 1.)

    @staticmethod
    def value(x, constant):
        """Overloaded value() method.
        """
        return np.full(x.shape, constant)

    @staticmethod
    def jacobian(x, constant):
        """Overloaded jacobian() method.
        """
        d_constant = np.full((len(x),), 1.)
        return np.array([d_constant]).transpose()

    def cdf(self, x):
        """Overloaded cdf() method.
        """
        return self.Constant * x

    def ppf(self, q):
        """Overloaded ppf() method.
        """
        return self.xmin + q * (self.xmax - self.xmin)

    def init_parameters(self, xdata, ydata, sigma):
        """Overloaded init_parameters() method.
        """
        self.set_parameter('Constant', np.mean(ydata))



class Line(FitModelBase):

    """Straight-line model.

    .. math::
      f(x; m, q) = mx + q
    """

    PARAMETER_NAMES = ('Intercept', 'Slope')
    PARAMETER_DEFAULT_VALUES = (1., 1.)
    DEFAULT_PLOTTING_RANGE = (0., 1.)

    @staticmethod
    def value(x, intercept, slope):
        """Overloaded value() method.
        """
        return intercept + slope * x

    @staticmethod
    def jacobian(x, intercept, slope):
        """Overloaded jacobian() method.
        """
        d_intercept = np.full((len(x),), 1.)
        d_slope = x
        return np.array([d_intercept, d_slope]).transpose()



class Gaussian(FitModelBase):

    """One-dimensional Gaussian model.

    .. math::
      f(x; A, \\mu, \\sigma) = A e^{-\\frac{(x - \\mu)^2}{2\\sigma^2}}
    """

    PARAMETER_NAMES = ('Amplitude', 'Peak', 'Sigma')
    PARAMETER_DEFAULT_VALUES = (1., 0., 1.)
    PARAMETER_DEFAULT_BOUNDS = ([0., -np.inf, 0], [np.inf] * 3)
    DEFAULT_PLOTTING_RANGE = (-5., 5.)
    SIGMA_TO_FWHM = 2.3548200450309493

    @staticmethod
    def value(x, amplitude, peak, sigma):
        """Overloaded value() method.
        """
        return amplitude * np.exp(-0.5 * ((x - peak)**2. / sigma**2.))

    @staticmethod
    def der_amplitude(x, amplitude, peak, sigma):
        """Return the amplitude derivative of the function, to be used in the
        calculation of the Jacobian.
        """
        return np.exp(-0.5 / sigma**2. * (x - peak)**2.)

    @staticmethod
    def der_peak(x, amplitude, d_amplitude, peak, sigma):
        """Return the peak derivative of the function, to be used in the
        calculation of the Jacobian.

        Note that we pass the pre-calculated values of the amplitude derivatives
        in order not to repeat the same calculation more times than strictly
        necessary.
        """
        return amplitude * d_amplitude * (x - peak) / sigma**2.

    @staticmethod
    def der_sigma(x, amplitude, d_amplitude, peak, sigma):
        """Return the sigma derivative of the function, to be used in the
        calculation of the Jacobian.

        Note that we pass the pre-calculated values of the amplitude derivatives
        in order not to repeat the same calculation more times than strictly
        necessary.
        """
        return amplitude * d_amplitude * (x - peak)**2. / sigma**3.

    @staticmethod
    def jacobian(x, amplitude, peak, sigma):
        """Overloaded jacobian() method.
        """
        d_amplitude = Gaussian.der_amplitude(x, amplitude, peak, sigma)
        d_peak = Gaussian.der_peak(x, amplitude, d_amplitude, peak, sigma)
        d_sigma = Gaussian.der_sigma(x, amplitude, d_amplitude, peak, sigma)
        return np.array([d_amplitude, d_peak, d_sigma]).transpose()

    def init_parameters(self, xdata, ydata, sigma):
        """Overloaded init_parameters() method.
        """
        self.set_parameter('Amplitude', np.max(ydata))
        self.set_parameter('Peak', np.mean(xdata))
        self.set_parameter('Sigma', np.std(xdata))

    def fwhm(self):
        """Return the absolute FWHM of the model.
        """
        return self.SIGMA_TO_FWHM * self.sigma

    def resolution(self):
        """Return the resolution of the model, i.e., the FWHM divided by the
        peak value.
        """
        if self.peak > 0:
            return self.fwhm() / self.peak
        return 0.

    def resolution_error(self):
        """Return the error on the resolution.
        """
        if self.peak > 0 and self.parameter_error('Sigma') > 0:
            return self.resolution() * self.parameter_error('Sigma') /\
                self.parameter_value('Sigma')
        return 0.



class PowerLaw(FitModelBase):

    """Power law model.

    .. math::
      f(x; N, \\Gamma) = N x^\\Gamma
    """

    PARAMETER_NAMES = ('Normalization', 'Index')
    PARAMETER_DEFAULT_VALUES = (1., -1.)
    PARAMETER_DEFAULT_BOUNDS = ([0., -np.inf], [np.inf, np.inf])
    DEFAULT_PLOTTING_RANGE = (1.e-2, 1.)

    @staticmethod
    def value(x, normalization, index):
        """Overloaded value() method.
        """
        return normalization * (x**index)

    @staticmethod
    def jacobian(x, normalization, index):
        """Overloaded jacobian() method.
        """
        d_normalization = (x**index)
        d_index = np.log(x) * normalization * (x**index)
        return np.array([d_normalization, d_index]).transpose()



class Exponential(FitModelBase):

    """Exponential model.

    .. math::
      f(x; N, \\alpha) = N e^{\\alpha x}
    """

    PARAMETER_NAMES = ('Normalization', 'Index')
    PARAMETER_DEFAULT_VALUES = (1., -1.)
    PARAMETER_DEFAULT_BOUNDS = ([0., -np.inf], [np.inf]*2)
    DEFAULT_PLOTTING_RANGE = (0., 1.)

    @staticmethod
    def value(x, normalization, exponent):
        """Overloaded value() method.
        """
        return normalization * np.exp(exponent * x)

    @staticmethod
    def jacobian(x, normalization, exponent):
        """Overloaded jacobian() method.
        """

        d_normalization = np.exp(exponent * x)
        d_exponent = normalization * x * np.exp(exponent * x)
        return np.array([d_normalization, d_exponent]).transpose()
