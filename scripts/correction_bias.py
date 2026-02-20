import numpy as np
from aptapy.hist import Histogram1d
from aptapy.models import Gaussian
from aptapy.plotting import plt
import xraydb

from hexsample.hexagon import HexagonalLayout
from hexsample.pipeline import simulate, reconstruct, calibrate
from hexsample.calibration import CalibrationMatrixGain
from hexsample.clustering import ClusteringNN
from hexsample.roi import Padding
from hexsample.fileio import ReconInputFile

energy = 6000
electrons = energy / xraydb.ionization_potential("Si")
snratio = 20
enc = electrons / snratio

print(f"Number of electrons: {electrons:.2f}")
print(f"ENC: {enc:.2f}")

# Create the gain matrix file with gain distributed uniformly between 0.9 and 1.1.
gain_path = "/home/augusto/hexsampledata/correction/gain_matrix.h5"
gain = CalibrationMatrixGain(304, 352)
gain.matrix = np.random.uniform(low=0.8, high=1.2, size=(352, 304))
# gain.matrix = np.random.triangular(left=0.4, mode=0.9, right=1.2, size=(352, 304))
gain.to_hdf5(gain_path)


N = 10000
simulation_path = "/home/augusto/hexsampledata/correction/simulation_rnd.h5"
simulate(
        num_events=N,
        output_file=str(simulation_path),
        beam="disk",
        radius=0.05,
        enc=enc * np.mean(gain.matrix), # We need to scale the ENC by the mean gain to get the same S/N
        zero_sup_threshold=0,
        readout_mode="rectangular",
        pitch=0.005,
        layout=HexagonalLayout.ODD_R,
        num_cols=304,
        num_rows=352,
        map_gain_file=gain_path,
        padding=Padding(2, 2, 2, 2),
)

calibrate(
    input_file=simulation_path,
    suffix="lsm",
    energy=energy,
    zero_sup_threshold=3*int(enc * np.mean(gain.matrix)), # We need to scale the zero suppression threshold by the mean gain to get the same S/N
    gain_calibration_method="lsm"
)

cal_rnd = CalibrationMatrixGain.from_hdf5("/home/augusto/hexsampledata/correction/simulation_rnd_lsm_gain.h5")
gain_uniform_path = "/home/augusto/hexsampledata/correction/gain_matrix_uniform.h5"
gain_uniform = CalibrationMatrixGain(304, 352, default=np.mean(gain.matrix))
gain_uniform.to_hdf5(gain_uniform_path)

simulation_path_uniform = "/home/augusto/hexsampledata/correction/simulation_uniform.h5"
simulate(
        num_events=N,
        output_file=str(simulation_path_uniform),
        beam="disk",
        radius=0.05,
        enc=enc * np.mean(gain_uniform.matrix), # We need to scale the ENC by the mean gain to get the same S/N
        zero_sup_threshold=0,
        readout_mode="rectangular",
        pitch=0.005,
        layout=HexagonalLayout.ODD_R,
        num_cols=304,
        num_rows=352,
        map_gain_file=gain_uniform_path,
        padding=Padding(2, 2, 2, 2),
)


# Calibrate the gain and compare the residuals between the two cases. We want to see if the
# residuals are compatible (depend only on S/N) or if the gain non-uniformity introduces
# additional bias.

calibrate(
    input_file=simulation_path_uniform,
    suffix="lsm",
    energy=energy,
    zero_sup_threshold=3*int(enc * np.mean(gain_uniform.matrix)), # We need to scale the zero suppression threshold by the mean gain to get the same S/N
    gain_calibration_method="lsm"
)

cal_uniform = CalibrationMatrixGain.from_hdf5("/home/augusto/hexsampledata/correction/simulation_uniform_lsm_gain.h5")


# plt.figure()
# plt.imshow(cal_uniform.matrix, vmin=0.8, vmax=1.2)
# plt.colorbar()

# plt.figure()
# plt.imshow(cal_rnd.matrix, vmin=0.8, vmax=1.2)
# plt.colorbar()

uniform_residuals = (cal_uniform.matrix[cal_uniform.hits > 0] - gain_uniform.matrix[cal_uniform.hits > 0]) / gain_uniform.matrix[cal_uniform.hits > 0]
rnd_residuals = (cal_rnd.matrix[cal_rnd.hits > 0] - gain.matrix[cal_rnd.hits > 0]) / gain.matrix[cal_rnd.hits > 0]

bins = np.linspace(min(uniform_residuals.min(), rnd_residuals.min()), max(uniform_residuals.max(), rnd_residuals.max()), 100)
bins = np.linspace(-0.1, 0.1, 200)
plt.figure()
uni = Histogram1d(bins, label="uniform gain").fill(uniform_residuals)
rnd = Histogram1d(bins, label="random gain").fill(rnd_residuals)
uni_model = Gaussian()
uni_model.fit_iterative(uni)
rnd_model = Gaussian()
uni.plot(label="uniform gain")
rnd.plot(label="random gain")
uni_model.plot(fit_output=True, label="uniform gain fit")
plt.legend()

bias = uni_model.mu.value
unbiased_gain = CalibrationMatrixGain(304, 352)
unbiased_gain.matrix = cal_rnd.matrix / (1 + bias)
unbiased_gain.to_hdf5("/home/augusto/hexsampledata/correction/simulation_rnd_lsm_gain_unbiased.h5")

recon_path = reconstruct(
        input_file=simulation_path,
        suffix="with_calibrated_gain_correction_lsm",
        zero_sup_threshold=3*int(enc * np.mean(unbiased_gain.matrix)), # We need to scale the zero suppression threshold by the mean gain to get the same S/N
        max_neighbors=6,
        pos_recon_algorithm="centroid",
        map_gain_file="/home/augusto/hexsampledata/correction/simulation_rnd_lsm_gain_unbiased.h5",
        padding=Padding(2, 2, 2, 2),
        )

no_cal = reconstruct(
        input_file=simulation_path,
        suffix="without_gain_correction",
        zero_sup_threshold=3*int(enc * np.mean(unbiased_gain.matrix)), # We need to scale the zero suppression threshold by the mean gain to get the same S/N
        max_neighbors=6,
        pos_recon_algorithm="centroid",
        map_gain_file=None,
        padding=Padding(2, 2, 2, 2),
        )

mc_cal = reconstruct(
        input_file=simulation_path,
        suffix="mc_correction",
        zero_sup_threshold=3*int(enc * np.mean(cal_uniform.matrix)), # We need to scale the zero suppression threshold by the mean gain to get the same S/N
        max_neighbors=6,
        pos_recon_algorithm="centroid",
        map_gain_file=gain_path,
        padding=Padding(2, 2, 2, 2),
        )



cal = ReconInputFile(recon_path)
no_cal = ReconInputFile(no_cal)
mc_cal = ReconInputFile(mc_cal)
cal_energy = cal.column("energy")
no_cal_energy = no_cal.column("energy")
mc_cal_energy = mc_cal.column("energy")

model_e = Gaussian()
model_mc = Gaussian()
energy_edges = np.linspace(min(no_cal_energy), max(no_cal_energy), 101)
cal_hist = Histogram1d(energy_edges)
no_cal_hist = Histogram1d(energy_edges)
mc_cal_hist = Histogram1d(energy_edges)
cal_hist.fill(cal_energy)
no_cal_hist.fill(no_cal_energy)
mc_cal_hist.fill(mc_cal_energy)
plt.figure()
cal_hist.plot(label="with calibrated gain correction (unbiased)")
no_cal_hist.plot(label="without gain correction")
mc_cal_hist.plot(label="mc correction")
model_e.fit_iterative(cal_hist, num_sigma_left=1.5, num_sigma_right=1.5)
model_mc.fit_iterative(mc_cal_hist, num_sigma_left=1.5, num_sigma_right=1.5)
model_e.plot(fit_output=True, label="with calibrated gain correction (unbiased) fit")
model_mc.plot(fit_output=True, label="mc correction fit")
plt.legend()


plt.show()