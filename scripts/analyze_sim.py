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
def analyze_sim(thick : int, noise : int, pitch : int) -> None:
    """Creates the energy histogram of all events for a certain thickness, enc
    and pitch of detector and readout. 

    Arguments
    ---------
    - thick : int 
        Thickness of silicon detector in mu m 
    - noise : int
        Noise og the detector readout in enc 
    - pitch : int
        Pitch og the detector readout in mu m   
    """
    thr = 2 * noise 
    file_path = f'/Users/chiara/hexsampledata/sim_{thick}um_{noise}enc_{pitch}pitch_recon_nn2_thr{thr}.h5'
    print(file_path)
    recon_file = ReconInputFile(file_path)
    #Constructing the 1px mask
    cluster_size = recon_file.column('cluster_size')
    mask = cluster_size < 2
    #Creating histograms - for all evts and for only evts with 1px
    energy_hist = create_histogram(recon_file, 'energy', binning = 100)
    energy_hist_1px = create_histogram(recon_file, 'energy', mask = mask, binning = 100)
    plt.figure()
    #fitted_model = fit_histogram(energy_hist, DoubleGaussian, show_figure = False)
    plt.title(fr'Energy histogram for t = {thick} $\mu$m, ENC = {noise}, pitch = {pitch} - 1px evts')
    plt.xlabel('Energy [eV]')
    fitted_model_1px = fit_histogram(energy_hist_1px, fit_model=DoubleGaussian, show_figure = True)
    plt.figure()
    cluster_size_hist = create_histogram(recon_file, 'cluster_size', binning = 100)
    cluster_size_hist.plot()
    recon_file.close()

    plt.show()

if __name__ == '__main__':
    THICKNESS = 500
    ENC = 40
    PITCH = 50
    analyze_sim(THICKNESS, ENC, PITCH, **vars(ANALYZESIM_ARGPARSER.parse_args()))
