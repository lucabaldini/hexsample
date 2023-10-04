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

"""Test suite for source.py
"""

from loguru import logger

import numpy as np

from hexsample.fitting import fit_gaussian_iterative
from hexsample.hist import Histogram1d, Histogram2d
from hexsample.plot import plt, setup_gca
from hexsample.source import PointBeam, DiskBeam, GaussianBeam
from hexsample.source import LineForest


def test_point_beam(x0 : float = 1., y0 : float = -1., num_photons : int = 1000):
    """Unit test for the point beam.
    """
    beam = PointBeam(x0, y0)
    x, y = beam.rvs(num_photons)
    assert np.allclose(x, np.full(num_photons, x0))
    assert np.allclose(y, np.full(num_photons, y0))

def test_disk_beam(radius : float = 0.1, num_photons : int = 1000000):
    """Unit test for the gaussian beam
    """
    beam = DiskBeam(radius=radius)
    x, y = beam.rvs(num_photons)
    binning = np.linspace(-1.5 * radius, 1.5 * radius, 100)
    plt.figure('Disk beam')
    Histogram2d(binning, binning).fill(x, y).plot()
    setup_gca(xlabel='x [cm]', ylabel='y [cm]')

def test_gaussian_beam(sigma=0.1, num_photons=1000000):
    """Test a gaussian beam
    """
    beam = GaussianBeam(sigma=sigma)
    x, y = beam.rvs(num_photons)
    binning = np.linspace(-5. * sigma, 5. * sigma, 100)
    plt.figure('Gaussian beam')
    Histogram2d(binning, binning).fill(x, y).plot()
    setup_gca(xlabel='x [cm]', ylabel='y [cm]')
    plt.figure('Gaussian beam x projection')
    hx = Histogram1d(binning).fill(x)
    hx.plot()
    model = fit_gaussian_iterative(hx, num_sigma_left=3., num_sigma_right=3.)
    model.plot()
    model.stat_box()
    plt.figure('Gaussian beam y projection')
    hy = Histogram1d(binning).fill(y)
    hy.plot()
    model = fit_gaussian_iterative(hy, num_sigma_left=3., num_sigma_right=3.)
    model.plot()
    model.stat_box()

def _test_forest(element, initial_level='K', num_events=100000, chisq_test=True):
    """Generic tes for a line forest.
    """
    # Create the forest.
    forest = LineForest(element, initial_level)
    logger.debug(forest)
    plt.figure(f'{element} {initial_level} line forest')
    forest.plot()
    if chisq_test:
        # Extract a bunch of random energies...
        energy = forest.rvs(num_events)
        # ... and do a chisquare test against the original line probabilities.
        values, counts = np.unique(energy, return_counts=True)
        for val, cnts in zip(values, counts):
            logger.debug(f'{val} eV -> {cnts} counts')
        p = counts / counts.sum()
        sigma = np.sqrt(counts) / counts.sum() * (1. - p)
        logger.debug(f'Forest energies: {forest._energies}')
        logger.debug(f'Forest probabilities: {forest._probs}')
        chi2 = (((forest._probs - p) / sigma)**2).sum()
        ndof = len(values) - 1
        logger.debug(f'Chisquare / ndof = {chi2} / {ndof}...')
        assert chi2 - ndof <= 5. * np.sqrt(2. * ndof)

def test_cu_k_forest():
    """Test the Cu K forest.
    """
    _test_forest('Cu')

def test_mn_k_forest():
    """Test the Cu K forest.

    Note we're not doing the chisquare test, here, as two of the lines have the
    same energy, and the thing would require extra code to deal with that.
    """
    _test_forest('Mn', chisq_test=False)




if __name__ == '__main__':
    test_disk_beam()
    test_gaussian_beam()
    test_cu_k_forest()
    test_mn_k_forest()
    plt.show()
