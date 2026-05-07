# Copyright (C) 2023--2026 the hexsample team.
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

"""Test suite for hexsample.pdf
"""

import numpy as np
from aptapy.plotting import plt

from hexsample import rng
from hexsample.pdf import SpectrumPDF

rng.initialize()

def test_unimodal_spectrum_pdf(size: int = 10000):
    """Test the SpectrumPDF class with a unimodal distribution.
    """
    # Create a unimodal distribution.
    vals = rng.generator.normal(loc=1.0, scale=0.1, size=size)
    # Create the PDF and fit it to the data.
    pdf = SpectrumPDF()
    pdf.fit(vals)
    # Check that the mean of the PDF is close to the mean of the data.
    assert np.isclose(pdf.mean(), vals.mean(), atol=0.01)
    # Plot the pdf
    plt.figure("Unimodal distribution")
    xx = np.linspace(vals.min()*0.8, vals.max()*1.2, 1000)
    plt.plot(xx, pdf(xx), label="PDF")
    plt.hist(vals, bins=50, density=True, alpha=0.5, label="Data")
    plt.legend()


def test_bimodal_spectrum_pdf(size: int = 100000):
    """Test the SpectrumPDF class with a bimodal distribution.
    """
    # Extract values from a bimodal distribution.
    vals = rng.generator.choice((1., 2.), p=[0.7, 0.3], size=size)
    # Add some noise
    vals += rng.generator.normal(scale=0.2, size=size)
    # Create the PDF and fit it to the data.
    pdf = SpectrumPDF()
    pdf.fit(vals)
    # Check that the mean of the PDF is close to the mean of the data.
    assert np.isclose(pdf.mean(), vals.mean(), atol=0.01)
    # Plot the pdf
    plt.figure("Bimodal distribution")
    xx = np.linspace(vals.min()*0.8, vals.max()*1.2, 1000)
    plt.plot(xx, pdf(xx), label="PDF")
    plt.hist(vals, bins=50, density=True, alpha=0.5, label="Data")
    plt.legend()

def test_derivative(size: int = 100000):
    """Test the derivative of the PDF.
    """
    # Create a unimodal distribution.
    vals = rng.generator.normal(loc=1.0, scale=0.1, size=size)
    # Create the PDF and fit it to the data.
    pdf = SpectrumPDF()
    pdf.fit(vals)
    derivative = pdf.derivative
    # Check that the derivative of the PDF is close to zero at the mean of the data.
    plt.figure("Derivative of a gaussian PDF")
    xx = np.linspace(vals.min()*0.8, vals.max()*1.2, 10000)
    plt.plot(xx, derivative(xx))
    plt.legend()
