# Copyright (C) 2023--2025 the hexsample team.
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

"""Test suite for hexsample.resolution
"""

import numpy as np
from aptapy.plotting import plt

from hexsample import rng
from hexsample.resolution import SlantedEdgeResolution, SlitsAligner
from hexsample.source import SlitBeam

rng.initialize(seed=0)


def test_slits_aligner():
    """Test the SlitsAligner class.
    """
    theta = 10.
    x, y = SlitBeam(0., 0., 1., 100., theta).rvs(1000000)
    # Fit the angle of the slit
    aligner = SlitsAligner(0.05, 2.)
    aligner.align(x, y)
    theta_fit = np.abs(np.rad2deg(aligner.angle))
    assert np.isclose(theta_fit, theta, atol=0.1)


def test_slanted_edge(n: int = 1000000):
    """Test the SlantedEdgeResolution class.
    """
    # Generate a straight edge with some gaussian noise
    sigma = 0.05
    _, y = SlitBeam(0., 0., 1., 100., 0.).rvs(n)
    y += rng.generator.normal(loc=0, scale=sigma, size=n)
    # Initialize the class and plot the ESF
    resolution = SlantedEdgeResolution(y, 0.02, None)
    esf = resolution.esf
    plt.figure("test_esf")
    esf.plot()
    # Plot the LSF
    lsf = resolution.lsf
    plt.figure("test_lsf")
    lsf.plot()
    # Plot the MTF
    mtf, freq = resolution.mtf()
    plt.figure("test_mtf")
    plt.plot(freq, mtf, '.k')
    # Estimate the resolution and check that it is close to the true value
    sigma_fit = resolution.resolution
    assert np.isclose(sigma_fit, sigma, rtol=0.1)
