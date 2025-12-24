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

from dataclasses import dataclass, field
from typing import Tuple, Union

import matplotlib
import numpy as np
import xraydb
from aptapy.plotting import setup_gca

from . import rng
from .base import AbstractPlottable, AbstractRandomGenerator, type_proxy
from .hexagon import HexagonalGrid

__all__ = [
    "Line",
    "LineForest",
    "SpectrumProxy",
    "PointBeam",
    "DiskBeam",
    "GaussianBeam",
    "TriangularBeam",
    "HexagonalBeam",
    "BeamProxy",
    "Source",
]


class AbstractSpectrum(AbstractRandomGenerator, AbstractPlottable):

    """Abstract base class for a X-ray energy spectrum.

    Subclasses must implement the `rvs` (to generate random energies) and
    `render` (for plotting) methods.
    """

    pass


@dataclass
class Line(AbstractSpectrum):

    """Class describing a monochromatic emission line at a given energy.

    Arguments
    ---------
    energy : float
        The line energy in eV.
    """

    energy: float = 6000.0

    def rvs(self, size: int = 1) -> np.ndarray:
        """Overloaded method.
        """
        return np.full(size, self.energy)

    def render(self, axes: matplotlib.axes.Axes, **kwargs) -> None:
        """Overloaded method.
        """
        padding = 100.
        kwargs.setdefault("width", 0.001)
        kwargs.setdefault("color", "black")
        axes.bar(self.energy, 1., **kwargs)
        setup_gca(xlabel="Energy [eV]", ylabel="Relative intensity",
                  xmin=self.energy - padding, xmax=self.energy + padding, grids=True)


@dataclass
class LineForest(AbstractSpectrum):

    """Class describing a set of X-ray emission lines for a given element and
    initial level or excitation energy.

    The underlying implementation relies on the XrayDB package. See
    https://xraypy.github.io/XrayDB/python.html#x-ray-emission-lines
    for more information.

    Arguments
    ---------
    element : int or str
        atomic number or atomic symbol for the given element (default: "Cu").

    initial_level : str, optional
        IUPAC symbol of the initial level (default: "K"). Note that the Python interface
        to XrayDB allows specifying either the initial level or the excitation energy
        (in eV), with the latter superseding the former, as it means "all initial levels
        with below this energy". In this initial implementation we only support specifying
        the initial level, and support for excitation energy may be added in the future,
        if truly needed.
    """

    element: Union[str, int] = "Cu"
    initial_level: str = "K"

    def __post_init__(self) -> None:
        """Post-initialization.
        """
        # Retrieve all the X-ray lines for the given element and setup.
        self.line_dict = xraydb.xray_lines(self.element, self.initial_level, None)
        # Cache the line energies (in eV) and the corresponding probabilities...
        self._energies = np.array([line.energy for line in self.line_dict.values()])
        self._probs = np.array([line.intensity for line in self.line_dict.values()])
        # ... and normalize the probabilities to one.
        self._probs /= self._probs.sum()

    def rvs(self, size: int  = 1) -> np.ndarray:
        """Overloaded method.
        """
        return rng.generator.choice(self._energies, size, replace=True, p=self._probs)

    def render(self, axes: matplotlib.axes.Axes, **kwargs) -> None:
        """Overloaded method.
        """
        kwargs.setdefault("width", 0.01)
        kwargs.setdefault("color", "black")
        axes.bar(self._energies, self._probs, **kwargs)
        for x, y, name in zip(self._energies, self._probs, self.line_dict.keys()):
            label = f"{name} ({y:.2e} @ {x:.0f} eV)"
            axes.text(x, 1.2 * y, label, ha="center", size="small")
        setup_gca(xlabel="Energy [eV]", ylabel="Relative intensity", logy=True, grids=True)


@type_proxy
class SpectrumProxy:

    """Type proxy for the available spectrum types.
    """

    _KEY = "spectrum"
    _PROXY_DICT = {
        "line": Line,
        "forest": LineForest,
    }


class AbstractBeam(AbstractRandomGenerator):

    """Abstract base class for all the X-ray beam shapes.

    Subclasses must implement the `rvs` method.
    """

    pass


@dataclass
class PointBeam(AbstractBeam):

    """Point-like X-ray beam.

    Arguments
    ---------
    x0 : float
        The x-coordinate of the beam centroid in cm.

    y0 : float
        The y-coordinate of the beam centroid in cm.
    """

    x0: float = 0.
    y0: float = 0.

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Overloaded method.
        """
        x = np.full(size, self.x0)
        y = np.full(size, self.y0)
        return x, y


@dataclass
class DiskBeam(AbstractBeam):

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

    x0: float = PointBeam.x0
    y0: float = PointBeam.y0
    radius: float = 0.1

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Overloaded method.
        """
        # pylint: disable=invalid-name
        r = self.radius * np.sqrt(rng.generator.uniform(size=size))
        theta = rng.generator.uniform(0., 2. * np.pi, size=size)
        x = self.x0 + r * np.cos(theta)
        y = self.y0 + r * np.sin(theta)
        return x, y


@dataclass
class GaussianBeam(AbstractBeam):

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

    x0: float = PointBeam.x0
    y0: float = PointBeam.y0
    sigma: float = 0.1

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Overloaded method.
        """
        x = rng.generator.normal(self.x0, self.sigma, size=size)
        y = rng.generator.normal(self.y0, self.sigma, size=size)
        return x, y


@dataclass
class TriangularBeam(AbstractBeam):

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

    x0: float = PointBeam.x0
    y0: float = PointBeam.y0
    v1: Tuple[float, float] = (1., 0.)
    v2: Tuple[float, float] = (0., 1.)

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Overloaded method.
        """
        if len(self.v1) != 2 or len(self.v2) != 2:
            raise ValueError("v1 and v2 must have 2 elements.")

        v0_ar = np.array([self.x0, self.y0])
        v1_ar = np.array(self.v1)
        v2_ar = np.array(self.v2)

        u = rng.generator.uniform(0, 1, (size, 2))
        mask = u[:, 0] + u[:, 1] > 1
        u[mask, :] = 1 - u[mask, :]

        w = (v1_ar - v0_ar) * u[:, 0, None] + (v2_ar - v0_ar) * u[:, 1, None] + v0_ar
        return w[:, 0], w[:, 1]


@dataclass
class HexagonalBeam(AbstractBeam):

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

    x0: float = PointBeam.x0
    y0: float = PointBeam.y0
    v0: Tuple[float, float] = (1., 0.)
    v1: Tuple[float, float] = (0., 1.)

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Overloaded method.
        """
        _, size_t = np.unique(rng.generator.integers(0, 6, size), return_counts=True)
        x = np.zeros(size)
        y = np.zeros(size)

        j = 0
        c = np.array([self.x0, self.y0])
        for i, t_s in enumerate(size_t):
            rotator = HexagonalGrid.create_rotator(np.pi / 3. * i)
            v0_rot = rotator((self.v0[0] - c[0], self.v0[1] - c[1])) + c
            v1_rot = rotator((self.v1[0] - c[0], self.v1[1] - c[1])) + c
            beam = TriangularBeam(self.x0, self.y0, tuple(v0_rot), tuple(v1_rot))
            x_tr, y_tr = beam.rvs(t_s)

            x[j:j + t_s] = x_tr
            y[j:j + t_s] = y_tr
            j += t_s

        return x, y


@type_proxy(default="gaussian")
class BeamProxy:

    """Type proxy for the available beam types.
    """

    _KEY = "beam"
    _PROXY_DICT = {
        "point": PointBeam,
        "disk": DiskBeam,
        "gaussian": GaussianBeam,
        "triangular": TriangularBeam,
        "hexagonal": HexagonalBeam,
    }


@dataclass
class Source:

    """Class describing a fully-fledged X-ray source.

    Arguments
    ---------
    spectrum : AbstractSpectrum
        The source spectrum.

    beam : AbstractBeam
        The source morphology.

    rate : float
        The photon rate in Hz.
    """

    spectrum: AbstractSpectrum = field(default_factory=SpectrumProxy.default)
    beam: AbstractBeam = field(default_factory=BeamProxy.default)
    rate: float = 100.

    @classmethod
    def from_kwargs(cls, **kwargs) -> "Source":
        """Alternative constructor to create source objects from specifications.

        Arguments
        ---------
        kwargs : dict
            The keyword arguments containing the source specifications.

        Returns
        -------
        Source
            The source object.
        """
        spectrum = SpectrumProxy.factory(**kwargs)
        beam = BeamProxy.factory(**kwargs)
        rate = kwargs.get("rate", cls.rate)
        return cls(spectrum, beam, rate)

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract random X-ray initial properties.

        Arguments
        ---------
        size : int
            The number of X-rays to be generated.

        Returns
        -------
        t, energy, x, y : 4-element tuple of np.ndarray of shape ``size``
            The X-ray properties.
        """
        # pylint: disable=invalid-name
        tmin = 0.
        tmax = tmin + size / self.rate
        t = rng.generator.uniform(tmin, tmax, size)
        t.sort()
        energy = self.spectrum.rvs(size)
        x, y = self.beam.rvs(size)
        return t, energy, x, y
