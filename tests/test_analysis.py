# Copyright (C) 2023 luca.baldini@pi.infn.it, c.tomaiuolo@studenti.unipi.it
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
import numpy as np

from hexsample.analysis import create_histogram, fit_histogram
from hexsample.fileio import DigiInputFile, ReconInputFile
from hexsample.modeling import Gaussian
from hexsample.pipeline import hxsim, hxrecon
from hexsample.plot import plt


def test_histograms(num_events : int = 1000):
    """
    """
    digi_file_path = hxsim(numevents=num_events)
    recon_file_path = hxrecon(infile=digi_file_path)
    digi_file = DigiInputFile(digi_file_path)
    recon_file = ReconInputFile(recon_file_path)
    plt.figure('Energy')
    hist = create_histogram(recon_file, 'energy', mc=True)
    hist.plot(label='Monte Carlo')
    hist = create_histogram(recon_file, 'energy')
    hist.plot(label='Recon')
    plt.figure('Energy k_alpha')
    energy = recon_file.column('energy')
    mask = energy < 8500
    hist = create_histogram(recon_file, 'energy', mask=mask)
    recon_file.close()
    digi_file.close()
    hist.plot()

    #Is it fine to test the fit inside test_histograms or is better to create another test?

def test_fit_histogram(num_events : int = 1000):
    """
    """
    digi_file_path = hxsim(numevents=num_events)
    recon_file_path = hxrecon(infile=digi_file_path)
    digi_file = DigiInputFile(digi_file_path)
    recon_file = ReconInputFile(recon_file_path)
    hist = create_histogram(recon_file, 'energy')
    plt.figure('Fitted energy - DoubleGaussian')
    fitstatus = fit_histogram(hist, show_figure = True)
    energy = recon_file.column('energy')
    mask = energy < 8500
    hist = create_histogram(recon_file, 'energy', mask=mask)
    fitstatus = fit_histogram(hist, fit_model=Gaussian, p0=np.array([1., 8000., 150.]), show_figure = False)
    print(f'A second fit has been done with a Gaussian FitModelBase but not plotted. Parameters are: \n{fitstatus}')
    recon_file.close()
    digi_file.close()



if __name__ == '__main__':
    test_histograms()
    test_fit_histogram()
    plt.show()
