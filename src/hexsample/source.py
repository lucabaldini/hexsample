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
    "SpectrumType",
    "PointBeam",
    "DiskBeam",
    "GaussianBeam",
    "TriangularBeam",
    "HexagonalBeam",
    "BeamType",
    "Source",
]


class AbstractSpectrum(AbstractRandomGenerator, AbstractPlottable):

    """Abstract base class for a X-ray energy spectrum.
    """

    pass


@dataclass(frozen=True)
class LineSpec:

    """Specifications for a monochromatic emission line at a given energy.
    """

    energy: float = 6000.0


class Line(LineSpec, AbstractSpectrum):

    """Class describing a monochromatic emission line at a given energy.
    """

    def rvs(self, size: int = 1) -> np.ndarray:
        """Overloaded method.
        """
        return np.full(size, self.energy)

    def _render(self, axes: matplotlib.axes.Axes, **kwargs) -> None:
        """Overloaded method.
        """
        kwargs.setdefault("width", 0.001)
        kwargs.setdefault("color", "black")
        axes.bar(self.energy, 1., **kwargs)
        setup_gca(xlabel="Energy [eV]", ylabel="Relative intensity",
                  xmin=self.energy - 100., xmax=self.energy + 100., grids=True)


@dataclass(frozen=True)
class LineForestSpec:

    """Specifications for a set of X-ray emission lines for a given element and
    initial level.

    See https://xraypy.github.io/XrayDB/python.html#x-ray-emission-lines for
    more information.

    .. info::
        Note that the Python interface to XrayDB allows specifying either the
        initial level or the excitation energy (in eV), with the latter superseding
        the former, as it means "all initial levels with below this energy".

        In this initial implementation we only support specifying the initial level,
        and support for excitation energy may be added in the future, if needed.

    Arguments
    ---------
    element : int or str
        atomic number or atomic symbol for the given element (default: "Cu").

    initial_level : str, optional
        IUPAC symbol of the initial level (default: "K").
    """

    element: Union[str, int] = "Cu"
    initial_level: str = "K"


class LineForest(LineForestSpec, AbstractSpectrum):

    """Class describing a set of X-ray emission lines for a given element and
    initial level or excitation energy.
    """

    def __init__(self, element: Union[str, int], initial_level: str) -> None:
        """Constructor.
        """
        LineForestSpec.__init__(self, element, initial_level)
        # Retrieve all the X-ray lines for the given element and setup.
        self.line_dict = xraydb.xray_lines(element, initial_level, None)
        # Cache the line energies (in eV) and the corresponding probabilities...
        self._energies = np.array([line.energy for line in self.line_dict.values()])
        self._probs = np.array([line.intensity for line in self.line_dict.values()])
        # ... and make sure the probabilities are correctly normalized.
        self._probs /= self._probs.sum()

    def rvs(self, size: int  = 1) -> np.ndarray:
        """Overloaded method.
        """
        return rng.generator.choice(self._energies, size, replace=True, p=self._probs)

    def _render(self, axes: matplotlib.axes.Axes, **kwargs) -> None:
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
class SpectrumType:

    """Type proxy for the available spectrum types.
    """

    _KEY = "spectrum"

    _PROXY_DICT = {
        "line": Line,
        "forest": LineForest,
    }


class AbstractBeam(AbstractRandomGenerator):

    """Abstract base class for all the X-ray beam shapes.
    """

    pass


@dataclass(frozen=True)
class PointBeamSpec:

    """Specifications for a point-like X-ray beam.

    Arguments
    ---------
    x0 : float
        The x-coordinate of the beam centroid in cm.

    y0 : float
        The y-coordinate of the beam centroid in cm.
    """

    x0: float = 0.
    y0: float = 0.


class PointBeam(PointBeamSpec, AbstractBeam):

    """Point-like X-ray beam.
    """

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Overloaded method.
        """
        # pylint: disable=invalid-name
        x = np.full(size, self.x0)
        y = np.full(size, self.y0)
        return x, y


@dataclass(frozen=True)
class DiskBeamSpec:

    """Specifications for a uniform disk X-ray beam.

    Arguments
    ---------
    x0 : float
        The x-coordinate of the beam centroid in cm.

    y0 : float
        The y-coordinate of the beam centroid in cm.

    radius : float
        The disk radius in cm.
    """

    # pylint: disable=invalid-name
    x0: float = PointBeamSpec.x0
    y0: float = PointBeamSpec.y0
    radius: float = 0.1


class DiskBeam(DiskBeamSpec, AbstractBeam):

    """Uniform disk X-ray beam.
    """

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Overloaded method.
        """
        # pylint: disable=invalid-name
        r = self.radius * np.sqrt(rng.generator.uniform(size=size))
        theta = rng.generator.uniform(0., 2. * np.pi, size=size)
        x = self.x0 + r * np.cos(theta)
        y = self.y0 + r * np.sin(theta)
        return x, y


@dataclass(frozen=True)
class GaussianBeamSpec:

    """Specifications for an azimuthally-symmetric gaussian beam.

    Arguments
    ---------
    x0 : float
        The x-coordinate of the beam centroid in cm.

    y0 : float
        The y-coordinate of the beam centroid in cm.

    sigma : float
        The beam sigma in cm.
    """

    # pylint: disable=invalid-name
    x0: float = PointBeamSpec.x0
    y0: float = PointBeamSpec.y0
    sigma: float = 0.1


class GaussianBeam(GaussianBeamSpec, AbstractBeam):

    """Azimuthally-simmetric gaussian beam.
    """

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Overloaded method.
        """
        # pylint: disable=invalid-name
        x = rng.generator.normal(self.x0, self.sigma, size=size)
        y = rng.generator.normal(self.y0, self.sigma, size=size)
        return x, y


@dataclass(frozen=True)
class TriangularBeamSpec:

    """Specifications for a triangular uniform X-ray beam.

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

    # pylint: disable=invalid-name
    x0: float = PointBeamSpec.x0
    y0: float = PointBeamSpec.y0
    v1: Tuple[float, float] = (1., 0.)
    v2: Tuple[float, float] = (0., 1.)


class TriangularBeam(TriangularBeamSpec, AbstractBeam):

    """Triangular uniform X-ray beam.
    """

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


@dataclass(frozen=True)
class HexagonalBeamSpec:

    """Specifications for a hexagonal uniform X-ray beam.

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

    # pylint: disable=invalid-name
    x0: float = PointBeamSpec.x0
    y0: float = PointBeamSpec.y0
    v0: Tuple[float, float] = (1., 0.)
    v1: Tuple[float, float] = (0., 1.)


class HexagonalBeam(HexagonalBeamSpec, AbstractBeam):

    """Hexagonal uniform X-ray beam.
    """

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
class BeamType:

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


@dataclass(frozen=True)
class SourceSpec:

    """Specifications for a fully-fledged X-ray source.

    Arguments
    ---------
    spectrum : AbstractSpectrum
        The source spectrum.

    beam : AbstractBeam
        The source morphology.

    rate : float
        The photon rate in Hz.
    """

    spectrum: AbstractSpectrum = Line()
    beam: AbstractBeam = GaussianBeam()
    rate: float = 100.


class Source(SourceSpec):

    """Class describing a fully-fledged X-ray source.
    """

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
        spectrum = SpectrumType.factory(**kwargs)
        beam = BeamType.factory(**kwargs)
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
