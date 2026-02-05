import argparse
from pathlib import Path

import numpy as np
from aptapy.hist import Histogram1d, Histogram2d
from aptapy.plotting import plt

from hexsample.fileio import ReconInputFile
from hexsample.hexagon import HexagonalGrid, HexagonalLayout
from hexsample.pipeline import reconstruct, simulate
from hexsample.resolution import eef, hew

__description__ = ""

# Parser object.
HXETA_ARGPARSER = argparse.ArgumentParser(description=__description__)
HXETA_ARGPARSER.add_argument("enc", type=int, help="equivalent noise charge in electrons")
HXETA_ARGPARSER.add_argument("zero_sup_threshold", type=int, help="zero suppression threshold in electrons")
HXETA_ARGPARSER.add_argument("--save", action="store_true", help="save the figures")

RESOLUTION_DIR = Path.home() / "hexsampledata" / "resolution"
if not RESOLUTION_DIR.exists():
    RESOLUTION_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = Path.home() / "hexsample_figures" / "resolution"
if not FIGURES_DIR.exists():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _estimate_loc(hist: Histogram2d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the location of the distribution in each slice of the histogram.

    Arguments
    ---------
    hist : Histogram2d
        The 2D histogram to analyze.
    
    Returns
    -------
    centers : np.ndarray
        The bin centers of the first axis.
    loc : np.ndarray
        The estimated location of the distribution in each slice.
    err : np.ndarray
        The estimated uncertainty on the location in each slice.
    """
    centers = hist.bin_centers(axis=0)
    loc = np.zeros(centers.shape)
    err = np.zeros(centers.shape)
    for i in range(len(centers)):
        slice_ = hist.slice1d(i)
        n = slice_.content.sum()
        if n == 0:
            loc[i] = np.nan
            err[i] = np.nan
            continue
        # Estimate the location as the median of the distribution and its error
        x = np.repeat(slice_.bin_centers(), slice_.content.astype(int))
        loc[i] = np.mean(x)
        err[i] = np.std(x) / np.sqrt(n)  # Approximate std error of median
    # Remove NaN entries
    mask = ~np.isnan(loc)
    centers = centers[mask]
    loc = loc[mask]
    err = err[mask]
    return centers, loc, err


def resolution(**kwargs):
    enc = kwargs["enc"]
    zero_sup_threshold = kwargs["zero_sup_threshold"]
    file_prefix = f"simulation_resolution_{enc}enc_hexagonal"
    simulation_path = RESOLUTION_DIR / f"{file_prefix}.h5"
    # Run simulation if the output file does not already exist
    if not simulation_path.exists():
        simulate(
                num_events=10000,
                output_file=str(simulation_path),
                beam="hexagonal",
                enc=enc,
                zero_sup_threshold=0,
                readout_mode="circular",
                pitch=0.005,
                layout=HexagonalLayout.ODD_R,
                num_cols=304,
                num_rows=352,
                gain=1.,
        )
    # Reconstruct the simulated file with different algorithms
    centroid_prefix = f"recon_zsuprec{zero_sup_threshold}_centroid"
    centroid_path = RESOLUTION_DIR / f"{file_prefix}_{centroid_prefix}.h5"
    if not centroid_path.exists():
        reconstruct(
            input_file=str(simulation_path),
            suffix=centroid_prefix,
            zero_sup_threshold=zero_sup_threshold,
            max_neighbors=6,
            pos_recon_algorithm="centroid"
        )
    # All pixels reconstructed with the best algorithm (eta for 2 and 3, centroid otherwise)
    best_prefix = f"recon_zsuprec{zero_sup_threshold}_best"
    best_path = RESOLUTION_DIR / f"{file_prefix}_{best_prefix}.h5"
    if not best_path.exists():
        reconstruct(
            input_file=str(simulation_path),
            suffix=best_prefix,
            zero_sup_threshold=zero_sup_threshold,
            max_neighbors=6,
            pos_recon_algorithm="eta"
        )
    # Study the EEF
    x = np.linspace(0, 0.6, 101)
    xlabel = "Distance residual [pitch normalized]"
    ylabel = "Encircled Energy Fraction"
    # Plot the centroid eef for all cluster sizes
    centroid_recon_file = ReconInputFile(str(centroid_path))
    plt.figure(f"centroid_eef_{enc}enc_{zero_sup_threshold}zsup")
    plt.plot(x, eef(x, centroid_recon_file, 0),
             label=f"1 pix (HEW={hew(centroid_recon_file, 0):.2f})",
             linestyle="-", color="black")
    plt.plot(x, eef(x, centroid_recon_file, 1),
             label=f"2 pix (HEW={hew(centroid_recon_file, 1):.2f})",
             linestyle="--", color="black")
    plt.plot(x, eef(x, centroid_recon_file, 2),
             label=f"3 pix (HEW={hew(centroid_recon_file, 2):.2f})",
             linestyle=":", color="black")
    plt.plot(x, eef(x, centroid_recon_file, max_neighbors=6),
             label=f"All pix (HEW={hew(centroid_recon_file, max_neighbors=6):.2f})",
             linestyle="-.", color="black")
    plt.xlim(x[0], x[-1])
    plt.ylim(0, 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    # Plot the eta + centroid eef for all cluster sizes
    best_recon_file = ReconInputFile(str(best_path))
    plt.figure(f"best_eef_{enc}enc_{zero_sup_threshold}zsup")
    plt.plot(x, eef(x, best_recon_file, 0),
             label=f"1 pix (HEW={hew(best_recon_file, 0):.2f})",
             linestyle="-", color="black")
    plt.plot(x, eef(x, best_recon_file, 1),
             label=f"2 pix (HEW={hew(best_recon_file, 1):.2f})",
             linestyle="--", color="black")
    plt.plot(x, eef(x, best_recon_file, 2),
             label=f"3 pix (HEW={hew(best_recon_file, 2):.2f})",
             linestyle=":", color="black")
    plt.plot(x, eef(x, best_recon_file, max_neighbors=6),
             label=f"All pix (HEW={hew(best_recon_file, max_neighbors=6):.2f})",
             linestyle="-.", color="black")
    plt.xlim(x[0], x[-1])
    plt.ylim(0, 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    centroid_recon_file.close()
    best_recon_file.close()

    # Now we study the dependence of the resolution on the recon distance from the pixel center
    # For two pixel events eta and centroid
    # hist_2d_2pix_eta = Histogram2d(
    #     dr_binning, dr_binning,
    #     xlabel="Reconstructed distance from pixel center [pitch normalized]",
    #     ylabel="Distance residual [pitch normalized]")
    # hist_2d_2pix_eta.fill(dr0_2pix_eta, dr_2pix_eta)
    # plt.figure("2pix_eta_residuals_vs_dr0")
    # hist_2d_2pix_eta.plot()
    # hist_2d_2pix_centroid = Histogram2d(
    #     dr_binning, dr_binning,
    #     xlabel="Reconstructed distance from pixel center [pitch normalized]",
    #     ylabel="Distance residual [pitch normalized]")
    # hist_2d_2pix_centroid.fill(dr0_2pix_centroid, dr_2pix_centroid)
    # plt.figure("2pix_centroid_residuals_vs_dr0")
    # hist_2d_2pix_centroid.plot()
    # # Take the slices and calculate the mean and stddev
    # centers_2pix_eta, loc_2pix_eta, err_2pix_eta = _estimate_loc(hist_2d_2pix_eta)
    # centers_2pix_centroid, loc_2pix_centroid, err_2pix_centroid = _estimate_loc(hist_2d_2pix_centroid)
    # plt.figure("2pix_resolution_distance_dependency")
    # plt.errorbar(
    #     centers_2pix_eta, loc_2pix_eta, yerr=err_2pix_eta,
    #     fmt=".r", label="2pix ETA")
    # plt.errorbar(
    #     centers_2pix_centroid, loc_2pix_centroid, yerr=err_2pix_centroid,
    #     fmt=".b", label="2pix CENTROID")
    # plt.xlabel("Reconstructed distance from pixel center [pitch normalized]")
    # plt.ylabel("Mean distance residual [pitch normalized]")
    # plt.legend()
    
    # # For three pixel events eta and centroid
    # hist_2d_3pix_eta = Histogram2d(
    #     dr_binning, dr_binning,
    #     xlabel="Reconstructed distance from pixel center [pitch normalized]",
    #     ylabel="Distance residual [pitch normalized]")
    # hist_2d_3pix_eta.fill(dr0_3pix_eta, dr_3pix_eta)
    # plt.figure("3pix_eta_residuals_vs_dr0")
    # hist_2d_3pix_eta.plot()
    # hist_2d_3pix_centroid = Histogram2d(
    #     dr_binning, dr_binning,
    #     xlabel="Reconstructed distance from pixel center [pitch normalized]",
    #     ylabel="Distance residual [pitch normalized]")
    # hist_2d_3pix_centroid.fill(dr0_3pix_centroid, dr_3pix_centroid)
    # plt.figure("3pix_centroid_residuals_vs_dr0")
    # hist_2d_3pix_centroid.plot()
    # # Take the slices and calculate the mean and stddev
    # centers_3pix_eta, loc_3pix_eta, err_3pix_eta = _estimate_loc(hist_2d_3pix_eta)
    # centers_3pix_centroid, loc_3pix_centroid, err_3pix_centroid = _estimate_loc(hist_2d_3pix_centroid)
    # plt.figure("3pix_resolution_distance_dependency")
    # plt.errorbar(
    #     centers_3pix_eta, loc_3pix_eta, yerr=err_3pix_eta,
    #     fmt=".r", label="3pix ETA")
    # plt.errorbar(
    #     centers_3pix_centroid, loc_3pix_centroid, yerr=err_3pix_centroid,
    #     fmt=".b", label="3pix CENTROID")
    # plt.xlabel("Reconstructed distance from pixel center [pitch normalized]")
    # plt.ylabel("Mean distance residual [pitch normalized]")
    # plt.legend()

    # if kwargs["save"]:
    #     fig_format = "png"
    #     fig_1pix.savefig(FIGURES_DIR / f"1pix_resolution_{enc}enc_{zero_sup_threshold}zsup.{fig_format}", format=fig_format)
    #     fig_2pix.savefig(FIGURES_DIR / f"2pix_resolution_{enc}enc_{zero_sup_threshold}zsup.{fig_format}", format=fig_format)
    #     fig_3pix.savefig(FIGURES_DIR / f"3pix_resolution_{enc}enc_{zero_sup_threshold}zsup.{fig_format}", format=fig_format)
    #     fig_allpix.savefig(FIGURES_DIR / f"allpix_resolution_{enc}enc_{zero_sup_threshold}zsup.{fig_format}", format=fig_format)


if __name__ == "__main__":
    resolution(**vars(HXETA_ARGPARSER.parse_args()))
    plt.show()
