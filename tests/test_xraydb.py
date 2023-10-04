
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

"""Test suite for the xraydb external package.
"""

import numpy as np
import xraydb

from hexsample import HEXSAMPLE_TEST_DATA, logger
from hexsample.plot import plt, setup_gca


def _load_nist_data(file_name):
    """Load the benchmark data downloaded from NIST to test the xraydb cross sections.
    """
    file_path = HEXSAMPLE_TEST_DATA / file_name
    logger.info(f'Loading NIST XCOM data from {file_path}...')
    energy, coh, incoh, photo, total = np.loadtxt(file_path, unpack=True)
    # Convert the energy from MeV to eV
    energy *= 1.e6
    logger.info('Done, {len(energy)} row(s) found.')
    return energy, coh, incoh, photo, total


def test_si():
    """Unit test for Si.
    """
    energy, coh, incoh, photo, total = _load_nist_data('nist_xcom_si.txt')
    plt.figure('Photon cross section in Si')
    for label, nist in zip(('coh', 'incoh', 'photo', 'total'), (coh, incoh, photo, total)):
        logger.info(f'Comparing {label}...')
        # For elements there are two different ways to retrieve the attenuation
        # coefficients---via mu_elam() or material_mu(). The second works for
        # compounds too.
        xrdb = xraydb.mu_elam('Si', energy, label)
        density = xraydb.atomic_density('Si')
        xrdb2 = xraydb.material_mu('Si', energy, density, label) / density
        assert np.allclose(xrdb, xrdb2)
        plt.plot(energy, nist, label=f'{label} (NIST)')
        plt.plot(energy, xrdb, label=f'{label} (xraydb)')
        delta = xrdb - nist
        frac_delta = delta / nist
        max_frac_delta = max(abs(frac_delta))
        logger.info(f'Fractional delta: {frac_delta}')
        logger.info(f'Maximum fractional delta: {max_frac_delta}')
        assert max_frac_delta < 0.001
    setup_gca(xlabel='Energy [eV]', ylabel='Attenuation [cm$^2$ g$^{-1}$]',
        logy=True, logx=True, legend=True, grids=True)
    plt.show()


def test_cdte():
    """Unit test for CdTe.

    Interesting---xraydb seems to be missing the CdTe line at 4341 eV, and there
    is no way of doing a sensible unit test, here.
    """
    energy, coh, incoh, photo, total = _load_nist_data('nist_xcom_cdte.txt')
    plt.figure('Photon cross section in CdTe')
    density = 5.85
    for label, nist in zip(('coh', 'incoh', 'photo', 'total'), (coh, incoh, photo, total)):
        logger.info(f'Comparing {label}...')
        xrdb = xraydb.material_mu('CdTe', energy, density, label) / density
        plt.plot(energy, nist, label=f'{label} (NIST)')
        plt.plot(energy, xrdb, label=f'{label} (xraydb)')
        delta = xrdb - nist
        frac_delta = delta / nist
        max_frac_delta = max(abs(frac_delta))
        logger.info(f'Fractional delta: {frac_delta}')
        logger.info(f'Maximum fractional delta: {max_frac_delta}')
    setup_gca(xlabel='Energy [eV]', ylabel='Attenuation [cm$^2$ g$^{-1}$]',
        logy=True, logx=True, legend=True, grids=True)
    plt.show()
