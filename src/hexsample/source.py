# Copyright (C) 2023--2025 the hexsample team.
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

"""X-ray source description.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import xraydb
from aptapy.plotting import plt, setup_gca

from . import rng
from .hexagon import HexagonalGrid


@dataclass
class BeamBase:

    """Base class describing the morphology of a X-ray beam.

    Arguments
    ---------
    x0 : float
        The x-coordinate of the beam centroid in cm.

    y0 : float
        The y-coordinate of the beam centroid in cm.
    """

    # pylint: disable=too-few-public-methods, invalid-name

    x0: float = 0.
    y0: float = 0.

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Do-nothing hook to generate random positions in the x-y plane.

        Arguments
        ---------
        size : int
            The number of X-ray photon positions to be generated.

        Returns
        -------
        x, y : 2-element tuple of np.ndarray of shape ``size``
            The photon positions on the x-y plane.
        """
        raise NotImplementedError



@dataclass
class PointBeam(BeamBase):

    """Point-like X-ray beam.

    Arguments
    ---------
    x0 : float
        The x-coordinate of the beam centroid in cm.

    y0 : float
        The y-coordinate of the beam centroid in cm.
    """

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Overloaded method.

        Arguments
        ---------
        size : int
            The number of X-ray photon positions to be generated.

        Returns
        -------
        x, y : 2-element tuple of np.ndarray of shape ``size``
            The photon positions on the x-y plane.
        """
        # pylint: disable=invalid-name
        x = np.full(size, self.x0)
        y = np.full(size, self.y0)
        return x, y



@dataclass
class DiskBeam(BeamBase):

    """Uniform disk X-ray beam.

    Arguments
    ---------
    x0 : float
        The x-coordinate of the beam centroid in cm.

    y0 : float
        The y-coordinate of the beam centroid in cm.

    radius : float
        The disk radius in cm.
    """

    radius: float = 0.1

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Overloaded method.

        Arguments
        ---------
        size : int
            The number of X-ray photon positions to be generated.

        Returns
        -------
        x, y : 2-element tuple of np.ndarray of shape ``size``
            The photon positions on the x-y plane.
        """
        # pylint: disable=invalid-name
        r = self.radius * np.sqrt(rng.generator.uniform(size=size))
        theta = rng.generator.uniform(0., 2. * np.pi, size=size)
        x = self.x0 + r * np.cos(theta)
        y = self.y0 + r * np.sin(theta)
        return x, y


@dataclass
class GaussianBeam(BeamBase):

    """Azimuthally-simmetric gaussian beam.

    Arguments
    ---------
    x0 : float
        The x-coordinate of the beam centroid in cm.

    y0 : float
        The y-coordinate of the beam centroid in cm.

    sigma : float
        The beam sigma in cm.
    """

    sigma: float = 0.1

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Overloaded method.

        Arguments
        ---------
        size : int
            The number of X-ray photon positions to be generated.

        Returns
        -------
        x, y : 2-element tuple of np.ndarray of shape ``size``
            The photon positions on the x-y plane.
        """
        # pylint: disable=invalid-name
        x = rng.generator.normal(self.x0, self.sigma, size=size)
        y = rng.generator.normal(self.y0, self.sigma, size=size)
        return x, y


@dataclass
class TriangularBeam(BeamBase):
    """Triangular uniform X-ray beam. 

    Arguments
    ---------
    x0 : float
        The x-coordinate of the first vertex of the triangle in cm.
    
    y0 : float
        The y-coordinate of the first vertex of the triangle in cm.
    
    v1 : Tuple[float, float]
        The (x, y) coordinates of the second vertex of the triangle in cm.
    
    v2 : Tuple[float, float]
        The (x, y) coordinates of the third vertex of the triangle in cm.
    """

    v1: Tuple[float, float] = (1., 0.)
    v2: Tuple[float, float] = (0., 1.)

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Overloaded method.

        Arguments
        ---------
        size : int
            The number of X-ray photon positions to be generated.

        Returns
        -------
        x, y : 2-element tuple of np.ndarray of shape ``size``
            The photon positions on the x-y plane.
        """
        if len(self.v1) != 2 or len(self.v2) != 2:
            raise ValueError("v1 and v2 must have 2 elements.")

        v0_ar = np.array([self.x0, self.y0])
        v1_ar = np.array(self.v1)
        v2_ar = np.array(self.v2)

        u = rng.generator.uniform(0, 1, (size, 2))
        mask = u[:, 0] + u[:, 1] > 1
        u[mask, :] = 1 - u[mask, :]

        w = (v1_ar - v0_ar)*u[:, 0, None] + (v2_ar - v0_ar)*u[:, 1, None] + v0_ar
        return w[:, 0], w[:, 1]


@dataclass
class HexagonalBeam(BeamBase):
    """Hexagonal uniform X-ray beam. 

    Arguments
    ---------
    x0 : float
        The x-coordinate of the center of the hexagon in cm.

    y0 : float
        The y-coordinate of the center of the hexagon in cm.

    v0 : Tuple[float, float]
        The (x, y) coordinates of the first vertex of the hexagon in cm.

    v1 : Tuple[float, float]
        The (x, y) coordinates of the second vertex of the hexagon in cm.

    """
    v0: Tuple[float, float] = (1., 0.)
    v1: Tuple[float, float] = (0., 1.)

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Overloaded method.

        Arguments
        ---------
        size : int
            The number of X-ray photon positions to be generated.

        Returns
        -------
        x, y : 2-element tuple of np.ndarray of shape ``size``
            The photon positions on the x-y plane.
        """
        _, size_t = np.unique(rng.generator.integers(0, 6, size), return_counts=True)
        x = np.zeros(size)
        y = np.zeros(size)

        j = 0
        c = np.array([self.x0, self.y0])
        for i, t_s in enumerate(size_t):
            rotator = HexagonalGrid.create_rotator(np.pi/3*i)
            v0_rot = rotator((self.v0[0] - c[0], self.v0[1] - c[1])) + c
            v1_rot = rotator((self.v1[0] - c[0], self.v1[1] - c[1])) + c
            beam = TriangularBeam(self.x0, self.y0, tuple(v0_rot), tuple(v1_rot))
            x_tr, y_tr = beam.rvs(t_s)

            x[j:j+t_s] = x_tr
            y[j:j+t_s] = y_tr
            j += t_s

        return x, y


class SpectrumBase:

    """Base class for a photon energy spectrum.
    """

    def rvs(self, size: int = 1) -> np.ndarray:
        """Do-nothing hook to generate random energies.

        Arguments
        ---------
        size : int
            The number of X-ray energies to be generated.

        Returns
        -------
        energy : np.ndarray of shape ``size``
            The photon energies in eV.
        """
        raise NotImplementedError

    def plot(self) -> None:
        """Do-nothing plotting hook.
        """
        raise NotImplementedError



class LineForest(SpectrumBase):

    """Class describing a set of X-ray emission lines for a given element and
    initial level or excitation energy.

    See https://xraypy.github.io/XrayDB/python.html#x-ray-emission-lines for
    more information.

    Arguments
    ---------
    element : int or str
        atomic number or atomic symbol for the given element

    initial_level : str, optional
        iupac symbol of the initial level

    excitation_energy : float, optional
        excitation energy in eV
    """

    def __init__(self, element: Union[str, int], initial_level: Optional[str] = None,
                 excitation_energy: Optional[float] = None) -> None:
        """Constructor.
        """
        super().__init__()
        # Retrieve all the X-ray lines for the given element and setup.
        self.line_dict = xraydb.xray_lines(element, initial_level, excitation_energy)
        # Cache the line energies (in eV) and the corresponding probabilities...
        self._energies = np.array([line.energy for line in self.line_dict.values()])
        self._probs = np.array([line.intensity for line in self.line_dict.values()])
        # ... and make sure the probabilities are correctly normalized.
        self._probs /= self._probs.sum()

    def rvs(self, size: int  = 1) -> np.ndarray:
        """Throw random energies from the line forest.

        Arguments
        ---------
        size : int
            The number of X-ray energies to be generated.

        Returns
        -------
        energy : np.ndarray of shape ``size``
            The photon energies in eV.
        """
        return rng.generator.choice(self._energies, size, replace=True, p=self._probs)

    def plot(self) -> None:
        """Plot the line forest.
        """
        # pylint: disable=invalid-name
        plt.bar(self._energies, self._probs, width=0.0001, color="black")
        for x, y, name in zip(self._energies, self._probs, self.line_dict.keys()):
            label = f"{name} ({y:.2e} @ {x:.0f} eV)"
            plt.text(x, 1.2 * y, label, ha="center", size="small")
        setup_gca(xlabel="Energy [eV]", ylabel="Relative intensity", logy=True, grids=True)

    def __str__(self):
        """String formatting.
        """
        return f"{self.line_dict}"


class Line(SpectrumBase):
    """Class describing a monochromatic emission line at a given energy
    """

    def __init__(self, energy : float) -> None:
        """Constructor of the class
        """
        self._energy = energy
        self._prob = 1

    def rvs(self, size: int = 1) -> np.ndarray:
        """Generate energies at a fixed value.

        Arguments
        ---------
        size : int, optional
            The number of X-ray photon energies to be generated. Defaults to 1.

        Returns
        -------
        energy : np.ndarray
            The photon energies in eV.
        """
        return np.full(size, self._energy)

    def plot(self) -> None:
        """Plot the monochromatic line
        """
        plt.bar(self._energy, self._prob, width=0.001, color='black')
        plt.xlabel('Energy [eV]')
        plt.ylabel('Relative intensity')
        plt.yscale('log')
        plt.grid()


class Source:

    """Base class for a X-ray source.

    Arguments
    ---------
    rate : float
        The photon rate in Hz.

    spectrum : SpectrumBase
        The source spectrum.

    beam : BeamBase
        The source beam morphology.
    """

    def __init__(self, spectrum: SpectrumBase, beam: BeamBase, rate: float = 100.) -> None:
        """Constructor.
        """
        self.spectrum = spectrum
        self.beam = beam
        self.rate = rate

    def rvs_timestamp(self, size: int = 1, tmin: float = 0.) -> np.ndarray:
        """Extract random times.

        Arguments
        ---------
        size : int
            The number of X-ray timestamps to be generated.

        Returns
        -------
        timestamp : np.ndarray of shape ``size``
            The photon timestamps in eV.
        """
        tmax = tmin + size / self.rate
        timestamp = rng.generator.uniform(tmin, tmax, size)
        timestamp.sort()
        return timestamp

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract random X-ray initial properties.

        Arguments
        ---------
        size : int
            The number of X-ray properties to be generated.

        Returns
        -------
        timestamp, energy, x, y : 4-element tuple of np.ndarray of shape ``size``
            The photon properties.
        """
        # pylint: disable=invalid-name
        timestamp = self.rvs_timestamp(size)
        energy = self.spectrum.rvs(size)
        x, y = self.beam.rvs(size)
        return timestamp, energy, x, y
