# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""Event file viewer.
"""
import numpy as np

from hexsample.app import ArgumentParser
from hexsample.fileio import ReconInputFile
from hexsample.modeling import DoubleGaussian
from hexsample.plot import plt
from hexsample.analysis import create_histogram, fit_histogram

__description__ = \
""" 
    Preliminary analysis of a recon file of simulations.
    This is done to analyze 

    The functions performs the fit with a DoubleGaussian model and 
    plots the histogram containing the fit results. 
    
"""

# Parser object.
ANALYZESIM_ARGPARSER = ArgumentParser(description=__description__)
def analyze_sim(thick : int, noise : int) -> None:
    """Creates the energy histogram of all events for a certain thickness and enc
    of detector and readout. 

    Arguments
    ---------
    - thick : int 
        Thickness of silicon detector in mu m 
    - noise : int
        Noise og the detector readout in enc    
    """
    thr = 2 * noise
    file_path = f'/Users/chiara/hexsampledata/hxsim_recon.h5'
    #file_path = f'/Users/chiara/hexsampledata/sim_HexagonalLayout.ODD_Rum_0enc_srcsigma200um_recon.h5'
    #file_path = f'/Users/chiara/hexsampledata/sim_{thick}um_{noise}enc_recon_nn2_thr{thr}.h5'
    recon_file = ReconInputFile(file_path)
    #Constructing the 1px mask
    cluster_size = recon_file.column('cluster_size')
    mask = cluster_size < 2
    #Creating histograms - for all evts and for only evts with 1px
    energy_hist = create_histogram(recon_file, 'energy', binning = 100)
    energy_hist_1px = create_histogram(recon_file, 'energy', mask = mask, binning = 100)
    plt.figure()
    fitted_model = fit_histogram(energy_hist, DoubleGaussian, show_figure = True)
    plt.title(fr'Energy histogram for t = {thick} $\mu$m, ENC = {noise}')
    plt.xlabel('Energy [eV]')
    #fitted_model_1px = fit_histogram(energy_hist_1px, fit_model=DoubleGaussian, show_figure = True)

    plt.figure()
    x_hist = create_histogram(recon_file, 'absx', mc = True, binning = 100)
    x_hist.plot()
    plt.xlabel('x [cm]')

    plt.figure()
    y_hist = create_histogram(recon_file, 'absy', mc = True, binning = 100)
    y_hist.plot()
    plt.xlabel('y [cm]')
    recon_file.close()

    plt.show()

if __name__ == '__main__':
    THICKNESS = 350
    ENC = 0
    analyze_sim(THICKNESS, ENC, **vars(ANALYZESIM_ARGPARSER.parse_args()))
