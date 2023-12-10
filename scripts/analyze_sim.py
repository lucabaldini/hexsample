# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""Event file viewer.
"""

import numpy as np

from hexsample.app import ArgumentParser
from hexsample.modeling import FitModelBase
from hexsample.fitting import fit_gaussian_iterative
from hexsample.hist import Histogram1d
from hexsample.fileio import ReconInputFile
from hexsample.modeling import Gaussian, DoubleGaussian
from hexsample.plot import plt
from hexsample.analysis import create_histogram, fit_histogram, double_heatmap

__description__ = \
""" 
    Preliminary analysis of a recon file of simulations.
    This is done to analyze 

    The functions performs the fit with a DoubleGaussian model and 
    plots the histogram containing the fit results. 
    
"""

# Parser object.
ANALYZESIM_ARGPARSER = ArgumentParser(description=__description__)

def analyze_sim(thickness : int, enc : int, **kwargs) -> None:
    thr = 2 * enc
    file_path = f'/Users/chiara/hexsampledata/sim_{thickness}um_{enc}enc_recon_nn2_thr{thr}.h5'
    recon_file = ReconInputFile(file_path)
    #Constructing the 1px mask 
    cluster_size = recon_file.column('cluster_size')
    mask = cluster_size < 2
    #Creating histograms - for all evts and for only evts with 1px
    energy_hist = create_histogram(recon_file, 'energy', binning = 100)
    energy_hist_1px = create_histogram(recon_file, 'energy', mask = mask, binning = 100)
    print(f'The number of events is: {energy_hist.content.sum()}')
    plt.figure()
    fitted_model = fit_histogram(energy_hist, DoubleGaussian, show_figure = True)
    plt.title(fr'Energy histogram for t = {thickness} $\mu$m, ENC = {enc}')
    plt.xlabel('Energy [eV]')
    fitted_model_1px = fit_histogram(energy_hist_1px, DoubleGaussian, show_figure = False)
    recon_file.close()

    plt.show()


if __name__ == '__main__':
    thickness = 500
    enc = 40
    analyze_sim(thickness, enc, **vars(ANALYZESIM_ARGPARSER.parse_args()))