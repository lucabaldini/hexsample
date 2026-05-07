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

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde


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
        y_grid = self.pdf(x_grid)
        area = np.trapezoid(y_grid, x_grid)
        normalized_pdf = y_grid / area
        mean = np.sum(x_grid * normalized_pdf) / np.sum(normalized_pdf)
        return mean

    def to_file(self, file_path: str) -> str:
        """Save the PDF to a file as a numpy archive.
        
        Arguments
        ---------
        file_path : str
            The path to the file where the PDF will be saved.
        """
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

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the PDF at the given points.
        """
        if self.pdf is None:
            raise ValueError("The PDF is empty. Fit the PDF with data or load it from file.")
        return self.pdf(x)
