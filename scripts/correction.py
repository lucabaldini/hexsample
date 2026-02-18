import numpy as np
from aptapy.hist import Histogram1d
from aptapy.plotting import plt
import xraydb

from hexsample.hexagon import HexagonalLayout
from hexsample.pipeline import simulate, reconstruct, calibrate
from hexsample.calibration import CalibrationMatrixGain
from hexsample.clustering import ClusteringNN
from hexsample.roi import Padding
from hexsample.fileio import ReconInputFile



# Create the gain matrix file with gain distributed uniformly between 0.9 and 1.1.
gain_path = "/home/augusto/hexsampledata/correction/gain_matrix.h5"
gain = CalibrationMatrixGain(304, 352)
gain.matrix = np.random.uniform(low=0.9, high=1.1, size=(352, 304))
gain.to_hdf5(gain_path)


# Now simulate an event file with the gain matrix.
simulation_path = "/home/augusto/hexsampledata/correction/simulation.h5"
simulate(
        num_events=10000,
        output_file=str(simulation_path),
        beam="disk",
        radius=0.1,
        enc=20,
        zero_sup_threshold=0,
        readout_mode="rectangular",
        pitch=0.005,
        layout=HexagonalLayout.ODD_R,
        num_cols=304,
        num_rows=352,
        map_gain_file=gain_path,
        padding=Padding(2, 2, 2, 2),
)

# Now we calibrate the gain and try to reconstruct the energy spectrum
calibrate(
    input_file=simulation_path,
    suffix="lsm",
    energy=6000,
    zero_sup_threshold=20,
    gain_calibration_method="lsm"
)

calibrate(
    input_file=simulation_path,
    suffix="single",
    energy=6000,
    zero_sup_threshold=20,
    gain_calibration_method="single"
)



zero_sup_threshold = 40
# Now reconstruct the simulated file with and without gain correction, and compare the results.
reconstruct(
        input_file=simulation_path,
        suffix="with_gain_correction",
        zero_sup_threshold=zero_sup_threshold,
        max_neighbors=6,
        pos_recon_algorithm="centroid",
        map_gain_file=gain_path,
        padding=Padding(2, 2, 2, 2),
)

reconstruct(
        input_file=simulation_path,
        suffix="without_gain_correction",
        zero_sup_threshold=zero_sup_threshold,
        max_neighbors=6,
        pos_recon_algorithm="centroid",
        map_gain_file=None,
        padding=Padding(2, 2, 2, 2),
)

reconstruct(
        input_file=simulation_path,
        suffix="with_calibrated_gain_correction_lsm",
        zero_sup_threshold=zero_sup_threshold,
        max_neighbors=6,
        pos_recon_algorithm="centroid",
        map_gain_file="/home/augusto/hexsampledata/correction/simulation_lsm_gain.h5",
        padding=Padding(2, 2, 2, 2),
)

reconstruct(
        input_file=simulation_path,
        suffix="with_calibrated_gain_correction_single",
        zero_sup_threshold=zero_sup_threshold,
        max_neighbors=6,
        pos_recon_algorithm="centroid",
        map_gain_file="/home/augusto/hexsampledata/correction/simulation_single_gain.h5",
        padding=Padding(2, 2, 2, 2),
)

gain_calibrated_lsm = ReconInputFile("/home/augusto/hexsampledata/correction/simulation_with_calibrated_gain_correction_lsm.h5")
gain_calibrated_single = ReconInputFile("/home/augusto/hexsampledata/correction/simulation_with_calibrated_gain_correction_single.h5")
gain_corrected = ReconInputFile("/home/augusto/hexsampledata/correction/simulation_with_gain_correction.h5")
gain_uncorrected = ReconInputFile("/home/augusto/hexsampledata/correction/simulation_without_gain_correction.h5")

mc_energy = gain_corrected.mc_column("num_pairs") * xraydb.ionization_potential("Si")
corrected_energy = gain_corrected.column("energy")
uncorrected_energy = gain_uncorrected.column("energy")
calibrated_energy_lsm = gain_calibrated_lsm.column("energy")
calibrated_energy_single = gain_calibrated_single.column("energy")

energy_edges = np.linspace(min(uncorrected_energy), max(uncorrected_energy), 101)
corrected_hist = Histogram1d(energy_edges)
uncorrected_hist = Histogram1d(energy_edges)
mc_hist = Histogram1d(energy_edges)
mc_hist.fill(mc_energy)
corrected_hist.fill(corrected_energy)
uncorrected_hist.fill(uncorrected_energy)
calibrated_hist = Histogram1d(energy_edges)
calibrated_hist.fill(calibrated_energy_lsm)
calibrated_hist_single = Histogram1d(energy_edges)
calibrated_hist_single.fill(calibrated_energy_single)
plt.figure()
corrected_hist.plot(label="with gain correction")
uncorrected_hist.plot(label="without gain correction")
calibrated_hist.plot(label="with calibrated gain correction (LSM)")
calibrated_hist_single.plot(label="with calibrated gain correction (single)")
mc_hist.plot(label="MC")
plt.legend()
plt.show()