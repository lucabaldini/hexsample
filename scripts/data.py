
from asyncio.log import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
from aptapy.hist import Histogram1d, Histogram2d
from aptapy.models import Gaussian
from aptapy.plotting import plt
import xraydb

from hexsample.clustering import ClusteringNN
from hexsample.fileio import digi_input_file_class, peek_readout_type, DigiInputFileRectangular, ReconInputFile
from hexsample.hexagon import HexagonalLayout
from hexsample.readout import HexagonalReadoutCircular, HexagonalReadoutMode, HexagonalReadoutRectangular 
from hexsample.tasks import ReconstructionDefaults, reconstruct

DATA_INPUT_FILE_PATH = Path("/home/augusto/asix/hdf/020_0006531_data.h5")
SIM_FILE_PATH = Path("/home/augusto/asix/simulation/simulation_enc100_gain0p08_10kev_diffsigma60.h5")


def compare():
    # Run the reconstruction on the data and on the simulation
    pos_recon_algorithm = "eta"
    suffix = f"recon_zsup16_{pos_recon_algorithm}"
    data_recon_path = f"{DATA_INPUT_FILE_PATH.with_suffix('')}_{suffix}.h5"
    if not Path(data_recon_path).exists():
        reconstruct(str(DATA_INPUT_FILE_PATH), zero_sup_threshold=16,
                                 max_neighbors=6, suffix=suffix)
    sim_recon_path = f"{SIM_FILE_PATH.with_suffix('')}_{suffix}.h5"
    if not Path(sim_recon_path).exists():
        reconstruct(str(SIM_FILE_PATH), zero_sup_threshold=16, max_neighbors=6,
                            suffix=suffix)
    
    # cluster_size
    data_recon_file = ReconInputFile(data_recon_path)
    sim_recon_file = ReconInputFile(sim_recon_path)

    plt.figure()
    data_cluster_size = data_recon_file.column("cluster_size")
    sim_cluster_size = sim_recon_file.column("cluster_size")
    cluster_size_data_hist = Histogram1d(np.arange(0.5, 4.5, 1), xlabel="Cluster size")
    cluster_size_data_hist.fill(data_cluster_size)
    cluster_size_sim_hist = Histogram1d(np.arange(0.5, 4.5, 1), xlabel="Cluster size")
    cluster_size_sim_hist.fill(sim_cluster_size)
    cluster_size_data_hist.plot(alpha=0.5, label="Data")
    cluster_size_sim_hist.plot(alpha=0.5, label="Simulation")
    plt.legend()
    
    adc_counts_data = np.round(data_recon_file.column("energy") / xraydb.ionization_potential("Si")).astype(int)
    pha_hist_data = Histogram1d(np.arange(-0.5, 600.5, 1), xlabel="ADC")
    pha_hist_data.fill(adc_counts_data)
    plt.figure()
    pha_hist_data.plot(alpha=0.5, label="Data")

    adc_counts_sim = np.round(sim_recon_file.column("energy") / xraydb.ionization_potential("Si")).astype(int)
    pha_hist_sim = Histogram1d(np.arange(-0.5, 600.5, 1), xlabel="ADC")
    pha_hist_sim.fill(adc_counts_sim)
    pha_hist_sim.plot(alpha=0.5, label="Simulation")


    # plot recon positions
    x = data_recon_file.column("posx")
    y = data_recon_file.column("posy")
    plt.figure()
    plt.style.use("dark_background")
    plt.scatter(x, y, s=1, alpha=0.5, color='white', edgecolors='none', label="Data")
    data_recon_file.close()
    sim_recon_file.close()
    plt.tight_layout()
    plt.show()




def noise():
    # Note we cast the input file to string, in case it happens to be a pathlib.Path object.
    input_file_path = str(DATA_INPUT_FILE_PATH)
    if not input_file_path.endswith(".h5"):
        raise RuntimeError(f"Input file {input_file_path} does not look like a HDF5 file")

    # It is necessary to extract the reaodut type because every readout type
    # corresponds to a different DigiEvent type.
    readout_mode = peek_readout_type(input_file_path)
    # And we should get rid of all this crap when we store the readout type and all the
    # relevant metadata in the hdf5 file in a sensible way.
    file_type = digi_input_file_class(readout_mode)
    input_file = file_type(input_file_path)

    header = input_file.header
    args = HexagonalLayout(header["layout"]), header["num_cols"], header["num_rows"],\
        header["pitch"], header["enc"], header["gain"]
    if readout_mode is HexagonalReadoutMode.RECTANGULAR:
        readout = HexagonalReadoutRectangular(*args, padding=header["padding"])
    elif readout_mode is HexagonalReadoutMode.CIRCULAR:
        readout = HexagonalReadoutCircular(*args)
    else:
        raise RuntimeError(f"Unsupported readout mode: {readout_mode}")

    zero_sup_threshold = 16
    num_neighbors = 6
    clustering = ClusteringNN(readout, zero_sup_threshold, num_neighbors)

    noise_xedges = np.arange(-0.5, 100.5, 1)
    energy_xedges = np.arange(-0.5, 600.5, 1)
    cluster_size = np.arange(0.5, 4.5, 1)
    noise_hist = Histogram1d(noise_xedges, xlabel="ADC")
    energy_hist = Histogram1d(energy_xedges, xlabel="ADC")
    data_cluster_size_hist = Histogram1d(cluster_size, xlabel="Cluster size")
    simulation_cluster_size_hist = Histogram1d(cluster_size, xlabel="Cluster size")

    x = []
    y = []
    for _, event in tqdm(enumerate(input_file)):
        pha = event.pha.copy()
        max_ind = np.argmax(pha)
        max_ind = np.unravel_index(max_ind, pha.shape)
        pha[max_ind[0] - 1:max_ind[0] + 2, max_ind[1] - 1:max_ind[1] + 2] = 0
        pha = pha.flatten()
        noise_hist.fill(pha[pha > 0])

        try:
            cluster = clustering.run(event)
            energy_hist.fill(cluster.pha.sum())
            defaults = ReconstructionDefaults
            recon_dict = dict(
                eta_2pix_rad=defaults.eta_2pix_rad,
                eta_2pix_pivot=0,
                eta_3pix_rad0=defaults.eta_3pix_rad0,
                eta_3pix_rad1=defaults.eta_3pix_rad1,
                eta_3pix_rad_pivot=0,
                eta_3pix_theta0=defaults.eta_3pix_theta0,
                pitch=readout.pitch,
            )
            event_x, event_y = cluster.eta(**recon_dict)
            x.append(event_x)
            y.append(event_y)
            data_cluster_size_hist.fill(cluster.size())

        except IndexError as e:
            logger.warning(f"Error while clustering event {event.trigger_id}: {e}", exc_info=True)
            continue
        except ZeroDivisionError as e:
            # We can have events without any signal above the zero suppression threshold that trigger
            # zero division errors in the centroid
            # logger.warning(f"Error while clustering event {event.trigger_id}: {e}", exc_info=True)
            continue
    input_file.close()
    plt.figure()
    model = Gaussian()
    model.fit(noise_hist)
    noise_hist.plot()
    model.plot(fit_output=True)
    plt.legend()

    plt.figure()
    e_model = Gaussian()
    e_model.fit(energy_hist)
    energy_hist.plot()
    e_model.plot(fit_output=True)
    plt.legend()

    plt.figure()
    NBINS = 100
    xedges = np.linspace(min(x), max(x), NBINS)
    yedges = np.linspace(min(y), max(y), NBINS)
    pos_hist = Histogram2d(xedges, yedges)
    pos_hist.fill(x, y)
    pos_hist.plot()


    plt.figure()
    plt.scatter(x, y, s=1)

    plt.figure()
    data_cluster_size_hist.plot(alpha=0.5, label="Data")

    plt.show()




if __name__ == "__main__":
    # noise()
    compare()