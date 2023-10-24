# Copyright (C) 2022 luca.baldini@pi.infn.it
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Test suite for modeling.py
"""

from functools import partial

import numpy as np

from hexsample import rng
from hexsample.hist import Histogram1d
from hexsample.modeling import Constant, Line, Gaussian, PowerLaw, Exponential,\
    FitModelBase, DoubleGaussian, FitStatus
from hexsample.plot import plt, setup_gca

rng.initialize()


def test_fit_status():
    """
    """
    par_names = ('slope', 'intercept')
    par_values = (1., 1.)
    status = FitStatus(par_names, par_values, None)
    print()
    print(status)
    status.parameter_values = np.array([1.33, 0.55])
    status.covariance_matrix = np.array([[0.01, 0.], [0., 0.01]])
    status.chisquare = 21.3
    status.ndof = 16
    print(status)
    status.set_parameter_bounds('slope', 10., 20.)
    print(status)

def _test_model(model : FitModelBase, rvs : np.ndarray, p0=None, figname : str = None, **kwargs):
    """Basic test for a specific model.
    """
    hist = Histogram1d(np.linspace(rvs.min(), rvs.max(), 100)).fill(rvs)
    model.fit_histogram(hist, p0=p0)
    if figname is None:
        figname = f'{model.name()} fitting model'
    plt.figure(figname)
    hist.plot()
    model.plot()
    model.stat_box()
    num_sigma = (model.status.chisquare - model.status.ndof) / np.sqrt(2. * model.status.ndof)
    assert abs(num_sigma) < 5.
    setup_gca(xlabel='x [a. u.]', **kwargs)

def test_models():
    """Test a constant fit model.
    """
    _test_model(Constant(), rng.generator.uniform(-1., 1., size=100000))
    _test_model(Gaussian(), rng.generator.normal(size=100000))
    _test_model(PowerLaw(), rng.generator.power(2., size=100000))
    _test_model(Exponential(), rng.generator.exponential(2., size=100000), logy=True)
    rvs = np.append(rng.generator.normal(10., 1., size=100000),
        rng.generator.normal(15., 1., size=25000))
    _test_model(DoubleGaussian(), rvs, p0=(5000., 10., 1., 2500., 15., 1.))
    _test_model(Gaussian() + Gaussian(), rvs, p0=(5000., 10., 1., 2500., 15., 1.))

def test_bound_parameter():
    """Perform a simple fit with a bound on a parameter.
    """
    model = Gaussian()
    model.status.set_parameter_bounds('mean', -0.0001, 0.0001)
    _test_model(model, rng.generator.normal(size=100000), p0 = (1., 0., 1.), figname='Gaussian bounded')



if __name__ == '__main__':
    test_models()
    test_bound_parameter()
    plt.show()
