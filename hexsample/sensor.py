# Copyright (C) 2022--2023 luca.baldini@pi.infn.it
#
# For the license terms see the file LICENSE, distributed along with this
# software.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""Active sensor medium.
"""

from enum import Enum

import numpy as np
import scipy.stats
import xraydb

from hexsample import logger



class CrossSection(Enum):

    """Enum class expressing the various cross sections.
    """

    COHERENT = 'coh'
    INCOHERENT = 'incho'
    PHOTOELECTRIC = 'photo'
    TOTAL = 'total'



class Material:

    """Class describing a material.

    This will work for either an element or a compound, provided that the
    symbol is recognized by xraydb.
    """
    def __init__(self, symbol : str, fano_factor : float, density : float = None,
        ionization_potential : float = None) -> None:
        """Constructor.
        """
        self.symbol = symbol
        self.fano_factor = fano_factor
        self.density = density or xraydb.atomic_density(self.symbol)
        self.ionization_potential = ionization_potential or xraydb.ionization_potential(self.symbol)

    def _attenuation_length(self, energy : np.ndarray, kind : CrossSection) -> np.ndarray:
        """Return the attenuation length (in cm) for the material.

        Arguments
        ---------
        energy : array_like
            The energy (in eV) at which the attenuation length is calculated.

        kind : CrossSection
             The cross secttion to be used.
        """
        return 1. / xraydb.material_mu(self.symbol, energy, self.density, kind.value)

    def coherent_attenuation_length(self, energy : np.ndarray) -> np.ndarray:
        """Return the coherent attenuation length (in cm) for the material.

        Arguments
        ---------
        energy : array_like
            The energy (in eV) at which the attenuation length is calculated.
        """
        return self._attenuation_length(energy, CrossSection.COHERENT)

    def incoherent_attenuation_length(self, energy : np.ndarray) -> np.ndarray:
        """Return the incoherent attenuation length (in cm) for the material.

        Arguments
        ---------
        energy : array_like
            The energy (in eV) at which the attenuation length is calculated.
        """
        return self._attenuation_length(energy, CrossSection.INCOHERENT)

    def photoelectric_attenuation_length(self, energy : np.ndarray) -> np.ndarray:
        """Return the photoelectric attenuation length (in cm) for the material.

        Arguments
        ---------
        energy : array_like
            The energy (in eV) at which the attenuation length is calculated.
        """
        return self._attenuation_length(energy, CrossSection.PHOTOELECTRIC)

    def total_attenuation_length(self, energy : np.ndarray) -> np.ndarray:
        """Return the total length (in cm) for the material.

        Arguments
        ---------
        energy : array_like
            The energy (in eV) at which the attenuation length is calculated.
        """
        return self._attenuation_length(energy, CrossSection.TOTAL)

    def _mu_components(self, energy : np.ndarray, kind : CrossSection) -> dict:
        """Return the absorption coefficients (in cm^{-1}) for the various elements in a
        compound.

        Arguments
        ---------
        energy : array_like
            The energy (in eV) at which the attenuation length is calculated.

        kind : CrossSection
             The cross secttion to be used.
        """
        return xraydb.material_mu_components(self.symbol, energy, self.density, kind.value)

    def coherent_mu_components(self, energy : np.ndarray) -> np.ndarray:
        """Return the coherent absorption coefficients (in cm^{-1}) for the
        various elements in a compound.

        Arguments
        ---------
        energy : array_like
            The energy (in eV) at which the attenuation length is calculated.
        """
        return self._mu_components(energy, CrossSection.COHERENT)

    def incoherent_mu_components(self, energy : np.ndarray) -> np.ndarray:
        """Return the incoherent absorption coefficients (in cm^{-1}) for the
        various elements in a compound.

        Arguments
        ---------
        energy : array_like
            The energy (in eV) at which the attenuation length is calculated.
        """
        return self._mu_components(energy, CrossSection.INCOHERENT)

    def photoelectric_mu_components(self, energy : np.ndarray) -> np.ndarray:
        """Return the photoelectric absorption coefficients (in cm^{-1}) for the
        various elements in a compound.

        Arguments
        ---------
        energy : array_like
            The energy (in eV) at which the attenuation length is calculated.
        """
        return self._mu_components(energy, CrossSection.PHOTOELECTRIC)

    def total_mu_components(self, energy : np.ndarray) -> np.ndarray:
        """Return the total absorption coefficients (in cm^{-1}) for the
        various elements in a compound.

        Arguments
        ---------
        energy : array_like
            The energy (in eV) at which the attenuation length is calculated.
        """
        return self._mu_components(energy, CrossSection.TOTAL)

    def fluorescence_yield(self, edge : str, line : str, energy : np.ndarray) -> np.ndarray:
        """Return the fluorescence yield for an X-ray emission line or family of lines.

        Arguments
        ---------
        edge: str
            IUPAC symbol of X-ray edge.

        line : str
            Siegbahn notation for emission line.

        energy : array_like
            Incident X-ray energy in eV.
        """
        return xraydb.fluor_yield(self.symbol, edge, line, energy)

    def rvs_num_pairs(self, energy : np.ndarray) -> np.ndarray:
        """Extract the number of pairs for the primary ionization.
        """
        mean = energy / self.ionization_potential
        sigma = np.sqrt(mean * self.fano_factor)
        return np.round(np.random.normal(mean, sigma)).astype(int)

    def __str__(self):
        """String formatting.
        """
        return f'{self.symbol}, F = {self.fano_factor}, E_ion = {self.ionization_potential} eV'



# Definition of the active media of interest.
Silicon = Material('Si', 0.116)
Germanium = Material('Ge', 0.106)
CadmiumTelluride = Material('CdTe', fano_factor=0.15, density=5.85, ionization_potential=4.45)



class Sensor:

    """Simple class describing a sensor.

    This is essentially a parallel-plate like slab of material acting as an
    absorbing medium for impinging X-rays.

    Arguments
    ---------
    thickness : float
        The sensor thickness in cm.

    material : Material instance
        The sensor material.

    trans_diffusion_sigma : float
        The transverse diffusion sigma in um / sqrt(cm).
    """

    def __init__(self, thickness : float, material : Material, trans_diffusion_sigma : float) -> None:
        """Constructor.
        """
        self.thickness = thickness
        self.material = material
        self.trans_diffusion_sigma = trans_diffusion_sigma

    def photabsorption_efficiency(self, energy : np.ndarray) -> np.ndarray:
        """Return the photabsorption efficiency for a given array of energy values.
        """
        lambda_ = self.material.photoelectric_attenuation_length(energy)
        return 1. - np.exp(-self.thickness / lambda_)

    def rvs_absorption_depth(self, energy : np.ndarray) -> np.ndarray:
        """Exract random variates for the absorption depth.

        Note this is using a truncated exponential distribution with the maximum
        value corresponding to the thickness of the detector.
        """
        lambda_ = self.material.photoelectric_attenuation_length(energy)
        dist = scipy.stats.expon(scale=lambda_)
        return dist.ppf(np.random.uniform(0., dist.cdf(self.thickness)))



class SiliconSensor(Sensor):

    """Specialized class describing a silicon sensor.
    """

    def __init__(self, thickness : float = 0.03, trans_diffusion_sigma : float = 40.) -> None:
        """Constructor.
        """
        super().__init__(thickness, Silicon, trans_diffusion_sigma)
