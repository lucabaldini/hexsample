import numpy as np
from aptapy.hist import Histogram1d
from aptapy.plotting import plt
from aptapy.models import Gaussian
import xraydb

from hexsample.hexagon import HexagonalLayout
from hexsample.pipeline import simulate, reconstruct, calibrate
from hexsample.calibration import CalibrationMatrixGain
from hexsample.clustering import ClusteringNN
from hexsample.roi import Padding
from hexsample.fileio import ReconInputFile


energy = 6000
electrons = energy / xraydb.ionization_potential("Si")
snratio = 30
enc = electrons / snratio

print(f"Number of electrons: {electrons:.2f}")
print(f"ENC: {enc:.2f}")

# Create the gain matrix file with gain distributed uniformly between 0.9 and 1.1.
gain_path = "/home/augusto/hexsampledata/correction/gain_matrix.h5"
gain = CalibrationMatrixGain(304, 352)
gain.matrix = np.random.uniform(low=0.8, high=1.2, size=(352, 304))
# gain.matrix = np.random.triangular(left=0.4, mode=0.9, right=1.2, size=(352, 304))
gain.to_hdf5(gain_path)

ENC_GAIN = enc * np.mean(gain.matrix)

# Now simulate an event file with the gain matrix.
N = 10000
simulation_path = "/home/augusto/hexsampledata/correction/simulation_rnd.h5"
simulate(
        num_events=N,
        output_file=str(simulation_path),
        beam="disk",
        radius=0.05,
        enc=ENC_GAIN, # We need to scale the ENC by the mean gain to get the same S/N
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
    energy=energy,
    zero_sup_threshold=3*ENC_GAIN,
    gain_calibration_method="lsm"
)



recon_path = reconstruct(
        input_file=simulation_path,
        suffix="with_calibrated_gain_correction_lsm",
        zero_sup_threshold=3*ENC_GAIN, # We need to scale the zero suppression threshold by the mean gain to get the same S/N
        max_neighbors=6,
        pos_recon_algorithm="centroid",
        map_gain_file="/home/augusto/hexsampledata/correction/simulation_rnd_lsm_gain.h5",
        padding=Padding(2, 2, 2, 2),
        )


no_cal = reconstruct(
        input_file=simulation_path,
        suffix="without_gain_correction",
        zero_sup_threshold=3*ENC_GAIN, # We need to scale the zero suppression threshold by the mean gain to get the same S/N
        max_neighbors=6,
        pos_recon_algorithm="centroid",
        map_gain_file=None,
        padding=Padding(2, 2, 2, 2),
        )

cal = ReconInputFile(recon_path)
no_cal = ReconInputFile(no_cal)
cal_energy = cal.column("energy")
no_cal_energy = no_cal.column("energy")

model_e = Gaussian()
energy_edges = np.linspace(min(no_cal_energy), max(no_cal_energy), 101)
cal_hist = Histogram1d(energy_edges)
no_cal_hist = Histogram1d(energy_edges)
cal_hist.fill(cal_energy)
no_cal_hist.fill(no_cal_energy)
model_e.fit_iterative(cal_hist, num_sigma_left=1.5, num_sigma_right=1.5)
plt.figure()
cal_hist.plot(label="with calibrated gain correction (unbiased)")
no_cal_hist.plot(label="without gain correction")
model_e.plot(fit_output=True)
plt.legend()


plt.show()