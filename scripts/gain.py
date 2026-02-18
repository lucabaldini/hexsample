import numpy as np
from aptapy.plotting import plt
from aptapy.models import Gaussian
from aptapy.hist import Histogram1d, Histogram2d

from hexsample.calibration import CalibrationMatrixGain
from hexsample.fileio import DigiInputFileRectangular
from hexsample.hexagon import HexagonalGrid, HexagonalLayout
from hexsample.clustering import ClusteringNN
from tqdm import tqdm

input_file = DigiInputFileRectangular("/home/augusto/hexsampledata/correction/simulation.h5")
num_cols = input_file.header_value("num_cols")
num_rows = input_file.header_value("num_rows")
layout = HexagonalLayout.ODD_R

zero_sup_threshold = 2
grid = HexagonalGrid(layout, num_cols, num_rows)
clustering = ClusteringNN(grid, zero_sup_threshold=zero_sup_threshold, num_neighbors=6)

# Initialize the gain and noise matrices.
gain_lsm = CalibrationMatrixGain(num_cols, num_rows, 6000, method="lsm")
gain_single = CalibrationMatrixGain(num_cols, num_rows, 6000, method="single")

# gain_single.matrix = np.random.normal(loc=.08, scale=0.01, size=(num_rows, num_cols))
# gain_single.to_hdf5("/home/augusto/asix/test_flat_gain_gaussian_loc1_scale0p2.h5")
gain_input = CalibrationMatrixGain.from_hdf5("/home/augusto/asix/test_flat_gain_gaussian_loc0p08_scale0p01.h5")._matrix
# # Loop over the events and update the gain and noise matrices.
for i, event in tqdm(enumerate(input_file)):
    try:
        cluster = clustering.run(event)
    except IndexError as e:
        continue
    gain_lsm.analyze_cluster(cluster, grid)
    gain_single.analyze_cluster(cluster, grid)

    if i >= 100000:
        break
input_file.close()

print(gain_lsm.default, gain_single.default)
residuals_lsm = (gain_lsm.matrix - gain_input) / gain_input
residuals_single = (gain_single.matrix - gain_input) / gain_input
print(gain_lsm.default, gain_single.default)

# plt.figure()
# plt.imshow(gain_single.matrix, cmap="viridis")
# plt.colorbar()
N = 0
hist_lsm = residuals_lsm[gain_lsm.hits > N]
hist_single = residuals_single[gain_single.hits > 0]
# print(np.mean(gain_lsm.hits[gain_lsm.hits > 0]), np.mean(gain_single.hits[gain_single.hits > 0]))
print(len(hist_lsm), len(hist_single))

xbins = np.linspace(-0.08, 0.08, 100)
lsm_hist = Histogram1d(xbins)
lsm_hist.fill(hist_lsm)
model = Gaussian()
plt.figure()
model.fit(lsm_hist)
model.plot(fit_output=True)
lsm_hist.plot(label="LSM")
plt.hist(hist_single, bins=xbins, alpha=0.5, label="Single")
# plt.hist(hist_lsm, bins=xbins, alpha=0.5, label="LSM")
plt.legend()


plt.imshow

plt.show()




