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

"""Test suite for hist.py
"""

import numpy as np

from hexsample.hist import Histogram1d, Histogram2d
from hexsample.plot import plt



def test_hist1d():
    """
    """
    binning = np.linspace(-5., 5., 100)
    h = Histogram1d(binning, xlabel='x [a. u.]')
    h.fill(np.random.normal(size=1000000))
    plt.figure('1-dimensional histogram')
    h.plot()


def test_hist2d():
    """
    """
    binning = np.linspace(-5., 5., 100)
    plt.figure('2-dimensional histogram')
    h = Histogram2d(binning, binning, xlabel='x [a. u.]', ylabel='y [a. u.]')
    x = np.random.normal(size=1000000)
    y = np.random.normal(size=1000000)
    h.fill(x, y).plot()



if __name__ == '__main__':
    test_hist1d()
    test_hist2d()
    plt.show()
