"""Analyze eta function to reconstruct events position.
"""

from pathlib import Path

import numpy as np
from aptapy.plotting import plt

from hexsample.calibration import (
    angle,
    calibrate_dr_2pix,
    calibrate_dr_3pix,
    calibrate_theta_3pix,
    calibration_data,
    distance,
)
from hexsample.cli import argparse
from hexsample.clustering import ClusteringNN
from hexsample.fileio import digi_input_file_class, peek_readout_type
from hexsample.hexagon import HexagonalLayout
from hexsample.logging_ import logger
from hexsample.readout import HexagonalReadoutCircular, HexagonalReadoutMode

__description__ = \
"""Run the calibration of the eta function.
"""

# Parser object.
HXETA_ARGPARSER = argparse.ArgumentParser(description=__description__)
HXETA_ARGPARSER.add_argument("input_file", type=str, help="path to the input file")
HXETA_ARGPARSER.add_argument("zero_sup_threshold", type=int,
                             help="zero suppression threshold in electrons")
HXETA_ARGPARSER.add_argument("--save", action="store_true",
                            help="save the calibration plots to the results directory")


NUMBINS = 50
RESULTS_DIR = Path.home() / "hexsample_figures"
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def hxeta(**kwargs) -> tuple[float, float, float, float]:
    """Application to calibrate the eta function.
    """
    zero_sup_threshold = kwargs["zero_sup_threshold"]
    input_file_path = str(kwargs["input_file"])
    if not input_file_path.endswith(".h5"):
        raise RuntimeError(f"Input file {input_file_path} does not look like a HDF5 file")

    readout_mode = peek_readout_type(input_file_path)
    if readout_mode is not HexagonalReadoutMode.CIRCULAR:
        raise RuntimeError("Only CIRCULAR readout is supported.")
    file_type = digi_input_file_class(readout_mode)
    input_file = file_type(input_file_path)
    header = input_file.header
    args = HexagonalLayout(header["layout"]), header["num_cols"], header["num_rows"],\
        header["pitch"], header["enc"], header["gain"], zero_sup_threshold
    readout = HexagonalReadoutCircular(*args)
    nneighbors = 6
    logger.info(f"Readout chip: {readout}")
    clustering = ClusteringNN(readout, zero_sup_threshold, nneighbors)
    # Create all the lists we need to fill
    size, photon_pos, versors, eta = calibration_data(input_file, clustering, header["pitch"])
    input_file.close()

    plot_kwargs = dict(save=kwargs["save"], path=RESULTS_DIR)
    # 2-pixel events calibration
    mask_2pix = size == 2
    eta_2pix = eta[mask_2pix].flatten()
    dr_2pix = distance(photon_pos[mask_2pix], versors[mask_2pix, 0])
    sigma_2pix = calibrate_dr_2pix(eta_2pix, dr_2pix, nbins=NUMBINS, **plot_kwargs)

    # 3-pixel events calibration
    mask_3pix = size == 3
    eta_3pix = np.stack(eta[mask_3pix])
    dr_3pix = distance(photon_pos[mask_3pix])
    theta_3pix = angle(photon_pos[mask_3pix], versors[mask_3pix])
    offset_rad_3pix, sigma_rad_3pix = calibrate_dr_3pix(eta_3pix, dr_3pix, nbins=NUMBINS,
                                                        **plot_kwargs)
    sigma_theta_3pix = calibrate_theta_3pix(eta_3pix, dr_3pix, theta_3pix, nbins=NUMBINS,
                                            **plot_kwargs)

    return sigma_2pix, offset_rad_3pix, sigma_rad_3pix, sigma_theta_3pix

if __name__ == "__main__":
    hxeta(**vars(HXETA_ARGPARSER.parse_args()))
    plt.show()
