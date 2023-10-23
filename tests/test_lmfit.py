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

"""Random exploration of lmfit in view of its possible adoption.
"""

from lmfit.models import GaussianModel
import numpy as np

from hexsample import rng
from hexsample.hist import Histogram1d
from hexsample.plot import plt

rng.initialize()


def test_gaussian():
    """
    """
    rvs = rng.generator.normal(size=100000)
    hist = Histogram1d(np.linspace(rvs.min(), rvs.max(), 100)).fill(rvs)
    hist.plot()
    xdata, ydata, sigma = hist.bin_centers(), hist.content, hist.errors()
    mask = ydata > 0
    xdata = xdata[mask]
    ydata = ydata[mask]
    sigma = sigma[mask]
    model = GaussianModel()
    print(dir(model))
    model.print_param_hints()
    print(model.param_names)
    print(model.eval(x=0))
    result = model.fit(ydata, x=xdata, weights=1. / sigma)
    plt.plot(xdata, result.best_fit)
    print(result.fit_report())
    print(result.best_values)
    #plt.figure('plot')
    #result.plot_fit()


if __name__ == '__main__':
    test_gaussian()
    plt.show()
