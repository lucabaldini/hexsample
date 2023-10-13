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
from hexsample.sensor import Silicon, SiliconSensor


def test_efficiency():
    """Calculate the efficiency of a silicon detector for different thickness values.
    """
    plt.figure('Silicon efficiency')
    energy = np.linspace(2000., 15000., 200)
    for thickness in (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1):
        sensor = SiliconSensor(thickness)
        efficiency = sensor.photabsorption_efficiency(energy)
        plt.plot(energy, efficiency, label=f'{1.e4 * thickness:.0f} $\\mu$m')
    setup_gca(xlabel='Energy [eV]', ylabel='Photoabsorption efficiency',
        grids=True, legend=True, xmax=energy.max())

def test_attenuation_length():
    """Calculate the photoelectric attenuation length of a silicon detector.
    """
    plt.figure('Silicon attenuation length')
    energy = np.linspace(2000., 15000., 200)
    attenuation_length = Silicon.photoelectric_attenuation_length(energy)
    plt.plot(energy, attenuation_length)
    setup_gca(xlabel='Energy [eV]', ylabel='Photoelectric attenuation length [cm]',
        grids=True, logy=True, xmax=energy.max())

def test_absorption_depth(thickness=0.05, energy=8000., num_photons=100000):
    """Extract random absorption depths.
    """
    plt.figure('Absorption depth')
    sensor = SiliconSensor(thickness)
    _energy = np.full(num_photons, energy)
    d = sensor.rvs_absorption_depth(_energy)
    h = Histogram1d(np.linspace(0., 1.1 * thickness, 101)).fill(d)
    h.plot()
    setup_gca(xlabel='Absorption depth [cm]', logy=True)
    model = fit_histogram(Exponential(), h)
    model.plot()
    model.stat_box()
    index = model.parameter_value('Index')
    sigma_index = model.parameter_error('Index')
    lambda_ = -1. / index
    sigma_lambda = sigma_index / index**2.
    delta = (Silicon.photoelectric_attenuation_length(energy) - lambda_) / sigma_lambda
    assert delta < 5.



if __name__ == '__main__':
    test_efficiency()
    test_attenuation_length()
    test_absorption_depth()
    plt.show()
