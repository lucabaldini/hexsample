from hexsample.calibration import CalibrationMatrixNoise, CalibrationMatrixGain
from hexsample.fileio import DigiInputFileRectangular
from aptapy.plotting import plt
from aptapy.models import Gaussian
from tqdm import tqdm
import numpy as np
from aptapy.hist import Histogram1d

input_file = DigiInputFileRectangular("/home/augusto/asix/hdf/020_0006531_data_small.h5")

num_rows = input_file.header_value("num_rows")
num_cols = input_file.header_value("num_cols")

# matrix = CalibrationMatrixNoise(num_cols, num_rows, default=0.)
# gain_matrix = CalibrationMatrixGain(num_cols, num_rows, 9700, 16)
# for i, event in tqdm(enumerate(input_file)):
#     matrix += event
#     gain_matrix += event


# matrix.to_hdf5("/home/augusto/asix/test_noise_matrix.h5")
# gain_matrix.to_hdf5("/home/augusto/asix/test_gain_matrix.h5")
# input_file.close()

# matrix = CalibrationMatrixNoise.from_hdf5("/home/augusto/asix/test_noise_matrix.h5")
gain_matrix = CalibrationMatrixGain.from_hdf5("/home/augusto/asix/test_gain_matrix.h5")



# counts, bins = np.histogram(gain_matrix.value[gain_matrix.num_events > 0], bins=10)
# prob = counts / np.sum(counts)

# bin_centers = (bins[:-1] + bins[1:]) / 2
# sample = np.random.choice(bin_centers, size=1000, p=prob)

plt.imshow(gain_matrix.flat_field())
plt.colorbar()
# model = Gaussian()
# xedges = np.arange(len(matrix.histogram) + 1) - 0.5
# noise_hist = Histogram1d(xedges)
# noise_hist.set_content(matrix.histogram)

# plt.figure("Noise Histogram")
# noise_hist.plot()
# model.fit(noise_hist)
# model.plot(label="Gaussian fit", fit_output=True)
# plt.legend()

# plt.figure("Gain")
# plt.imshow(gain_matrix.value)
# plt.colorbar()

# plt.figure("Noise")
# plt.imshow(matrix.value, vmax=15)
# plt.colorbar()

# yy = gain_matrix.value.flatten()
# plt.figure("Gain Histogram")
# plt.hist(yy[yy > 0], bins=100)

# yy_noise = matrix.value.flatten()
# plt.figure("Noise 1d")
# plt.hist(yy_noise[yy_noise > 0], bins=100)


plt.show()