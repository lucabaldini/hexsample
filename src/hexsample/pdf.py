# Copyright (C) 2023--2026 the hexsample team.
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

"""Facilities to create and use probability density functions from spectra."""

from typing import Tuple

import numpy as np
from aptapy.plotting import plt
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde

try:
    from numpy import trapezoid as trapezoid
except ImportError:
    from numpy import trapz as trapezoid


class SpectrumPDF:

    """Create a probability density function from a spectrum.

    This class uses Gaussian kernel density estimation to create a smooth normalized
    probability density function (PDF) from a set of values. The PDF is then interpolated
    to allow for efficient evaluation.
    """

    def __init__(self) -> None:
        """Class constructor
        """
        self.pdf = None

    def fit(self, vals: np.ndarray) -> None:
        """Fit the data to create the PDF.

        Arguments
        ---------
        vals : np.ndarray
            The values to fit the PDF from.
        """
        # Use Gaussian kernel density estimation to create a smooth PDF.
        kde = gaussian_kde(vals)
        # Create a grid of points for interpolation. We need to ensure that the grid exceeds
        # the range of the data to avoid issues with extrapolation.
        x_grid = np.linspace(min(vals)*0.9, max(vals)*1.1, 1000)
        y_grid = kde(x_grid)
        self.pdf = CubicSpline(x_grid, y_grid, extrapolate=True)

    def mean(self) -> float:
        """Return the mean of the PDF.
        """
        if self.pdf is None:
            raise ValueError("The PDF is empty. Fit the PDF with data or load it from file.")
        # The location of the PDF can be estimated as the mean of the distribution.
        x_grid = np.linspace(min(self.pdf.x), max(self.pdf.x), 1000)
        # Clip the PDF to avoid negative values.
        y_grid = np.maximum(0, self.pdf(x_grid))
        area = trapezoid(y_grid, x_grid)
        mean = trapezoid(x_grid * y_grid, x_grid) / area
        return mean

    def to_file(self, file_path: str) -> str:
        """Save the PDF to a file as a numpy archive.
        
        Arguments
        ---------
        file_path : str
            The path to the file where the PDF will be saved.
        """
        if self.pdf is None:
            raise ValueError("The PDF is empty. Fit the PDF with data before saving it to file.")
        np.savez(file_path, x=self.pdf.x, c=self.pdf.c)
        return file_path

    @classmethod
    def from_file(cls, file_path: str) -> "SpectrumPDF":
        """Load the PDF from a numpy archive.
        
        Arguments
        ---------
        file_path : str
            The path to the file from which the PDF will be loaded.
        """
        if not file_path.endswith(".npz"):
            raise ValueError("The file must be a numpy archive with .npz extension.")
        data = np.load(file_path)
        instance = cls()
        instance.pdf = CubicSpline.construct_fast(data["c"], data["x"], extrapolate=True)
        return instance

    def derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        """Evaluate the derivative of the PDF at the given points.

        Arguments
        ---------
        x : np.ndarray
            The points at which to evaluate the derivative.
        order : int, optional
            The order of the derivative to evaluate (default is 1).
        """
        if self.pdf is None:
            raise ValueError("The PDF is empty. Fit the PDF with data or load it from file.")
        return self.pdf(x, nu=order)

    def plot(self, xlim: Tuple[float, float] = None, **kwargs) -> None:
        """Plot the PDF in a given range.

        Arguments
        ---------
        xlim : tuple[float, float], optional
            The range of x values to plot (default is None, which means the fitting range
            of the PDF).
        """
        plt.figure(kwargs.get("figname", "spectrum_pdf"))
        if xlim is None:
            xlim = (min(self.pdf.x), max(self.pdf.x))
        x_grid = np.linspace(xlim[0], xlim[1], 1000)
        y_grid = self.pdf(x_grid)
        plt.plot(x_grid, y_grid, **kwargs)
        plt.xlabel(kwargs.get("xlabel", "Energy [eV]"))
        plt.ylabel(kwargs.get("ylabel", "PDF"))
        plt.xlim(xlim)
        plt.grid(visible=True, linestyle="--", alpha=0.5, color="gray")
        plt.tight_layout()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the PDF at the given points.
        """
        if self.pdf is None:
            raise ValueError("The PDF is empty. Fit the PDF with data or load it from file.")
        return self.pdf(x)
