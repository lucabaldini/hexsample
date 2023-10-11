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

"""Test suite for sensor.py
"""

from loguru import logger

import numpy as np

from hexsample.fitting import fit_histogram
from hexsample.hist import Histogram1d
from hexsample.modeling import Exponential
from hexsample.plot import plt, setup_gca
from hexsample.sensor import SiliconSensor


def test_sensor():
    """
    """
    plt.figure('Efficiency')
    energy = np.linspace(1000., 20000., 200)
    for thickness in (0.01, 0.02, 0.03, 0.05, 0.075, 0.1):
        sensor = SiliconSensor(thickness)
        efficiency = sensor.photabsorption_efficiency(energy)
        plt.plot(energy, efficiency, label=f'{1.e4 * thickness} $\\mu$m')
    setup_gca(xlabel='Energy [eV]', ylabel='Photoabsorption efficiency',
        grids=True, legend=True)
    plt.figure('Absorption depth')
    sensor = SiliconSensor(thickness=0.03)
    energy = np.full(100000, 8000)
    d = sensor.rvs_absorption_depth(energy)
    h = Histogram1d(np.linspace(-0.01, 0.04, 101)).fill(d)
    h.plot()
    setup_gca(xlabel='Absorption depth [cm]', logy=True)
    model = fit_histogram(Exponential(), h)
    model.plot()
    model.stat_box()



if __name__ == '__main__':
    test_sensor()
    plt.show()
