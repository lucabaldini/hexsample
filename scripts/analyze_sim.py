# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""Event file viewer.
"""
import numpy as np
from scipy.stats import norm

from hexsample.app import ArgumentParser
from hexsample.fileio import ReconInputFile
from hexsample.modeling import FitStatus, DoubleGaussian
from hexsample.plot import plt
from hexsample.analysis import create_histogram, fit_histogram, Gini_index, energy_threshold_computation

__description__ = \
""" 
    Preliminary analysis of a recon file of simulations.
    This is done to analyze 

    The functions performs the fit with a DoubleGaussian model and 
    plots the histogram containing the fit results. 
    
"""

# Parser object.
ANALYZESIM_ARGPARSER = ArgumentParser(description=__description__)

def analyze_sim(thick: int, noise: int, pitch: int, contamination_beta_on_alpha: float=0.02) -> None:
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
    - contamination_beta_on_alpha : float
        Contamination of beta on alpha signal. It is a value in [0,1.)  
    """
    #Taking data from reconstructed simulations at chosen detector thickness and pitch and readout noise
    thr = 2 * noise 
    file_path = f'/Users/chiara/hexsampledata/sim_{thick}um_{noise}enc_{pitch}pitch_recon_nn2_thr{thr}.h5'
    print(file_path)
    recon_file = ReconInputFile(file_path)

    #Constructing 1px mask for further analysis
    energy = recon_file.column('energy')
    cluster_size = recon_file.column('cluster_size')
    mask = cluster_size < 2

    #Creating histograms - for all evts and for evts with signal on 1px
    energy_hist = create_histogram(recon_file, 'energy', binning = 100)
    energy_hist_1px = create_histogram(recon_file, 'energy', binning = 100)
    plt.figure()
    z = np.linspace(7000,10000,2000)
    plt.title(fr'Energy histogram for t = {thick} $\mu$m, ENC = {noise}, pitch = {pitch} - all evts')
    plt.xlabel('Energy [eV]')
    fitted_model = fit_histogram(energy_hist, DoubleGaussian, show_figure = True)
    #Doing the same for 1px evts
    '''
    plt.figure()
    plt.title(fr'Energy histogram for t = {thick} $\mu$m, ENC = {noise}, pitch = {pitch} - 1px evts')
    plt.xlabel('Energy [eV]')
    fitted_model_1px = fit_histogram(energy_hist_1px, fit_model=DoubleGaussian, show_figure = True)
    '''
    energy_thr, efficiency_of_alpha = energy_threshold_computation(fitted_model, contamination_beta_on_alpha)
    print(f"The energy thr corresponding to a contamination of beta on alpha signal\
           = {contamination_beta_on_alpha*100}% is {energy_thr}, with a corresponding\
           efficiency of alpha signal = {efficiency_of_alpha*100}%")
    #Plotting energy threshold vline on figure
    plt.axvline(energy_thr, color='r', label = rf'$C_{{\beta \rightarrow \alpha}} = {contamination_beta_on_alpha}$')
    
    '''
    #Finding contamination and efficiency using Gini coefficient
    #Constructing labels using mc column of energy.
    #Labels are: 0=Kalpha and 1=Kbeta
    #Labels for Gini computation
    data_labels = (recon_file.mc_column('energy') > 8200).astype(int)
    plt.vlines(energy_thr, 0, 6000, color='r', label = rf'$\epsilon_{{\beta}} = {efficiency}$')
    z_min = z[(z > 8046) & (z < 8900)]
    z_min = z_min[np.argmin(Gini_index(z_min, energy, data_labels))]
    print(z_min)
    plt.vlines(z_min, 0, 6000, color='g', label='Energy at min Gini idx')
    plt.legend(loc='upper left')
    plt.figure('Gini')
    plt.plot(z, Gini_index(z, energy, data_labels))
    '''

    '''
    #plt.figure()
    cluster_size_hist = create_histogram(recon_file, 'cluster_size', binning = 100)
    #cluster_size_hist.plot()
    recon_file.close()

    plt.show()
    print(f'Mean cluster size: {np.mean(cluster_size)}')
    '''
    recon_file.close()
    plt.show()

if __name__ == '__main__':
    THICKNESS = 500
    ENC = 40
    PITCH = 50
    contamination = 0.05
    analyze_sim(THICKNESS, ENC, PITCH, contamination)
