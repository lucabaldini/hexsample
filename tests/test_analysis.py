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

"""Test suite for hexsample.analysis
"""

from aptapy.plotting import plt

from hexsample.analysis import create_histogram
from hexsample.fileio import DigiInputFileRectangular, ReconInputFile
from hexsample.pipeline import hxrecon, hxsim


def test_histograms(num_events : int = 1000):
    """Test the histogram creation from recon files.
    """
    digi_file_path = hxsim(numevents=num_events)
    recon_file_path = hxrecon(infile=digi_file_path)
    digi_file = DigiInputFileRectangular(digi_file_path)
    recon_file = ReconInputFile(recon_file_path)
    plt.figure("Energy")
    hist = create_histogram(recon_file, "energy", mc=True)
    hist.plot(label="Monte Carlo")
    hist = create_histogram(recon_file, "energy")
    hist.plot(label="Recon")
    plt.figure("Energy k_alpha")
    energy = recon_file.column("energy")
    mask = energy < 8500
    hist = create_histogram(recon_file, "energy", mask=mask)
    recon_file.close()
    digi_file.close()
    hist.plot()
