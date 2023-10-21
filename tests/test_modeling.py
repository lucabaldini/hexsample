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

import numpy as np

from hexsample import rng
from hexsample.fitting import fit_histogram
from hexsample.hist import Histogram1d
from hexsample.modeling import Constant, Line, Gaussian, PowerLaw, Exponential,\
    FitModelBase, DoubleGaussian
from hexsample.plot import plt, setup_gca

rng.initialize()


def _test_model(model : FitModelBase, rvs : np.ndarray, p0=None, **kwargs):
    """Basic test for a specific model.
    """
    hist = Histogram1d(np.linspace(rvs.min(), rvs.max(), 100)).fill(rvs)
    fit_histogram(model, hist, p0=p0)
    plt.figure(f'{model.name()} fitting model')
    hist.plot()
    model.plot()
    model.stat_box()
    num_sigma = (model.chisq - model.ndof) / np.sqrt(2. * model.ndof)
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



if __name__ == '__main__':
    test_models()
    plt.show()
