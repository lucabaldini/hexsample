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

"""Test suite for source.py
"""

import numpy as np
from aptapy.hist import Histogram1d, Histogram2d
from aptapy.models import Gaussian
from aptapy.plotting import plt, setup_gca

from hexsample import rng
from hexsample.logging_ import logger
from hexsample.source import (
    DiskBeam,
    GaussianBeam,
    HexagonalBeam,
    Line,
    LineForest,
    PointBeam,
    Source,
    TriangularBeam,
)

rng.initialize()


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
    plt.figure("Disk beam")
    Histogram2d(binning, binning).fill(x, y).plot()
    setup_gca(xlabel="x [cm]", ylabel="y [cm]")


def test_gaussian_beam(sigma=0.1, num_photons=1000000):
    """Test a gaussian beam
    """
    beam = GaussianBeam(sigma=sigma)
    x, y = beam.rvs(num_photons)
    binning = np.linspace(-5. * sigma, 5. * sigma, 100)
    plt.figure("Gaussian beam")
    Histogram2d(binning, binning).fill(x, y).plot()
    setup_gca(xlabel="x [cm]", ylabel="y [cm]")
    plt.figure("Gaussian beam x projection")
    hx = Histogram1d(binning).fill(x)
    hx.plot()
    model = Gaussian()
    model.fit_iterative(hx, num_sigma_left=3., num_sigma_right=3.)
    model.plot(fit_output=True)
    plt.legend()

    plt.figure("Gaussian beam y projection")
    hy = Histogram1d(binning).fill(y)
    hy.plot()
    model = Gaussian()
    model.fit_iterative(hy, num_sigma_left=3., num_sigma_right=3.)
    model.plot(fit_output=True)
    plt.legend()


def test_triangular_beam(num_photons: int = 10000):
    """Test for TriangularBeam class
    """
    beam = TriangularBeam(0, 0, (0, 1), (1, 0))
    x, y = beam.rvs(num_photons)
    binning_x = np.linspace(min(x), max(x), 100)
    binning_y = np.linspace(min(y), max(y), 100)

    plt.figure("Triangular beam")
    Histogram2d(binning_x, binning_y).fill(x, y).plot()
    setup_gca(xlabel="x [cm]", ylabel="y [cm]")

    plt.figure("Triangular beam x projection")
    hx = Histogram1d(binning_x).fill(x)
    hx.plot()

    plt.figure("Triangular beam y projection")
    hy = Histogram1d(binning_y).fill(y)
    hy.plot()


def test_hexagonal_beam(size: int = 10000):
    """Test for HexagonalBeam class

    Args:
        size (int, optional): Number of photons to sample. Defaults to 10000.
    """
    beam = HexagonalBeam(0, 0, (1, 0), (.5, np.sqrt(3)/2))
    x, y = beam.rvs(size)
    binning_x = np.linspace(min(x), max(x), 100)
    binning_y = np.linspace(min(y), max(y), 100)

    plt.figure("Hexagonal beam")
    Histogram2d(binning_x, binning_y).fill(x, y).plot()
    setup_gca(xlabel="x [cm]", ylabel="y [cm]")

    plt.figure("Hexagonal beam x projection")
    hx = Histogram1d(binning_x).fill(x)
    hx.plot()

    plt.figure("Hexagonal beam y projection")
    hy = Histogram1d(binning_y).fill(y)
    hy.plot()


def _test_forest(element, initial_level="K", num_events=100000, chisq_test=True):
    """Generic test for a line forest.
    """
    # Create the forest.
    # pylint: disable=protected-access
    forest = LineForest(element, initial_level)
    logger.debug(forest)
    plt.figure(f"{element} {initial_level} line forest")
    forest.plot()
    if chisq_test:
        # Extract a bunch of random energies...
        energy = forest.rvs(num_events)
        # ... and do a chisquare test against the original line probabilities.
        values, counts = np.unique(energy, return_counts=True)
        for val, cnts in zip(values, counts):
            logger.debug(f"{val} eV -> {cnts} counts")
        p = counts / counts.sum()
        sigma = np.sqrt(counts) / counts.sum() * (1. - p)
        logger.debug(f"Forest energies: {forest._energies}")
        logger.debug(f"Forest probabilities: {forest._probs}")
        chi2 = (((forest._probs - p) / sigma)**2).sum()
        ndof = len(values) - 1
        logger.debug(f"Chisquare / ndof = {chi2} / {ndof}...")
        assert chi2 - ndof <= 5. * np.sqrt(2. * ndof)


def test_cu_k_forest():
    """Test the Cu K forest.
    """
    _test_forest("Cu", chisq_test=False)


def test_mn_k_forest():
    """Test the Mn K forest.

    Note we're not doing the chisquare test, here, as two of the lines have the
    same energy, and the thing would require extra code to deal with that.
    """
    _test_forest("Mn", chisq_test=False)


def test_line(energy: float = 6000, size: int = 10000):
    """Test for line class
    """
    line = Line(energy)
    logger.debug(line)
    x = line.rvs(size)

    values, counts = np.unique(x, return_counts=True)
    logger.debug(f"Beam energy: {energy}")
    logger.debug(f"Number of events: {size}")
    for val, cnts in zip(values, counts):
        logger.debug(f"{val} eV -> {cnts} counts")
        # Check if all events have the same energy given as input
        assert val == energy
        assert cnts == size


def test_source():
    """Test different ways to create a fully-fledged X-ray source.
    """
    source = Source(
        spectrum=Line(energy=8000.),
        beam=GaussianBeam(sigma=0.1),
        rate=5000.,
    )
    assert source.spectrum.energy == 8000.
    assert source.beam.x0 == 0.
    assert source.beam.y0 == 0.
    assert source.beam.sigma == 0.1
    assert source.rate == 5000.
