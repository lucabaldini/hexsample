import numpy as np
from aptapy.plotting import plt

from hexsample.fileio import ReconInputFile
from hexsample.resolution import SlantedEdgeResolution, SlitsAligner


def rotate():
    file_path_eta = "/home/augusto/asix/hdf/020_0006531_data_all_recon_zsup16_eta.h5"
    file_path_centroid = "/home/augusto/asix/hdf/020_0006531_data_all_recon_zsup16_centroid.h5"
    # file_path_eta = "/home/augusto/hexsampledata/edge_simulation_recon_eta.h5"
    # file_path_centroid = "/home/augusto/hexsampledata/edge_simulation_recon_centroid.h5"
    
    recon_file_eta = ReconInputFile(file_path_eta)
    x_eta = recon_file_eta.column("posx")
    y_eta = recon_file_eta.column("posy")

    recon_file_centroid = ReconInputFile(file_path_centroid)
    x_centroid = recon_file_centroid.column("posx")
    y_centroid = recon_file_centroid.column("posy")

    aligner = SlitsAligner(bin_size=0.001, sigma=10.)
    plt.figure("Edges")
    plt.imshow(aligner._detect_edges(x_eta, y_eta), aspect='auto')
    _ , y_rot_eta = aligner.align(x_eta, y_eta) # convert to microns
    _, y_rot_centroid = aligner.align(x_centroid, y_centroid)
    y_rot_eta *= 10000  # Convert to microns
    y_rot_centroid *= 10000

    mask_eta = np.full_like(y_rot_eta, True, dtype=bool)
    mask_centroid = np.full_like(y_rot_centroid, True, dtype=bool)
    YMIN = -530
    YMAX = -450
    mask_eta = (y_rot_eta > YMIN) & (y_rot_eta < YMAX)
    mask_centroid = (y_rot_centroid > YMIN) & (y_rot_centroid < YMAX)
    BIN_SIZE = 0.5    # microns
    SIGMA = 4       # bins
    slanted_edge_eta = SlantedEdgeResolution(y_rot_eta[mask_eta],
                                             bin_size=BIN_SIZE,
                                             sigma=SIGMA)
    plt.figure("ESF")
    slanted_edge_eta.esf.plot()
    plt.figure("LSF")
    slanted_edge_eta.lsf.plot()
    plt.figure("MTF")
    mtf, freq = slanted_edge_eta.mtf()

    plt.plot(freq*1000, mtf, '.k')

    slanted_edge_centroid = SlantedEdgeResolution(y_rot_centroid[mask_centroid],
                                                  bin_size=BIN_SIZE,
                                                  sigma=SIGMA)
    plt.figure("ESF")
    slanted_edge_centroid.esf.plot()
    plt.figure("LSF")
    slanted_edge_centroid.lsf.plot()
    plt.figure("MTF")
    mtf, freq = slanted_edge_centroid.mtf()
    plt.plot(freq*1000, mtf, '.r')

rotate()
plt.show()