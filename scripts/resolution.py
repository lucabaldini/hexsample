import argparse
from pathlib import Path

import numpy as np
from aptapy.hist import Histogram1d, Histogram2d
from aptapy.plotting import plt

from hexsample.fileio import ReconInputFile
from hexsample.hexagon import HexagonalGrid, HexagonalLayout
from hexsample.pipeline import reconstruct, simulate

__description__ = ""

# Parser object.
HXETA_ARGPARSER = argparse.ArgumentParser(description=__description__)
HXETA_ARGPARSER.add_argument("--enc", type=int, help="equivalent noise charge in electrons")
HXETA_ARGPARSER.add_argument("--zero_sup_threshold", type=int, help="zero suppression threshold in electrons")
HXETA_ARGPARSER.add_argument("--save", action="store_true", help="save the figures")

RESOLUTION_DIR = Path.home() / "hexsampledata" / "resolution"
if not RESOLUTION_DIR.exists():
    RESOLUTION_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = Path.home() / "hexsample_figures" / "resolution"
if not FIGURES_DIR.exists():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def distance(input_file_path: str, grid):
    pitch = grid.pitch
    input_file = ReconInputFile(input_file_path)
    # Monte Carlo true positions
    x_mc = input_file.mc_column("absx")
    y_mc = input_file.mc_column("absy")
    # Monte Carlo true pixel centers
    x0, y0 = grid.pixel_to_world(*grid.world_to_pixel(x_mc, y_mc))
    x = input_file.column("posx")
    y = input_file.column("posy")
    # Calulate the reconstructed position from the center of the pixel
    dr0 = np.sqrt((x - x0) ** 2 + (y - y0) ** 2) / pitch
    # Calculate the residual from the Monte Carlo true position
    dr = np.sqrt((x - x_mc) ** 2 + (y - y_mc) ** 2) / pitch
    input_file.close()

    return dr, dr0


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
    # Define the simulation file path
    enc = kwargs["enc"]
    zero_sup_threshold = kwargs["zero_sup_threshold"]
    if enc is None or zero_sup_threshold is None:
        raise RuntimeError("Both enc and zero_sup_threshold arguments are required.")
    zero_sup_thr_simulation = 0
    file_prefix = f"simulation_resolution_{enc}enc_zsupsim{zero_sup_thr_simulation}_hexagonal"
    simulation_path = RESOLUTION_DIR / f"{file_prefix}.h5"
    # Run simulation if the output file does not already exist
    if not simulation_path.exists():
        simulate(
                num_events=100000,
                output_file=str(simulation_path),
                beam="hexagonal",
                enc=enc,
                # This may be the cause of the resolution degradation. If in simulate it is set to 30,
                # then some events are reconstructed very badly. Furthermore, when setting it to 30,
                # if the reconstruction is done with a zero suppression threshold of 0, the results
                # are diffrent from the ones obtained with a threshold of 30, all without noise.
                zero_sup_threshold=zero_sup_thr_simulation,
                readout_mode="circular",
                pitch=0.005,
                layout=HexagonalLayout.ODD_R,
                num_cols=304,
                num_rows=352,
                gain=1.,
                random_seed=0
        )
    # Reconstruct the simulated file with different algorithms
    # We start with centroid for all pixels
    all_pixel_centroid_prefix = f"recon_zsuprec{zero_sup_threshold}_allpix_centroid"
    all_pixel_centroid = RESOLUTION_DIR / f"{file_prefix}_{all_pixel_centroid_prefix}.h5"
    if not all_pixel_centroid.exists():
        reconstruct(
            input_file=str(simulation_path),
            suffix=all_pixel_centroid_prefix,
            zero_sup_threshold=zero_sup_threshold,
            max_neighbors=6,
            pos_recon_algorithm="centroid"
        )
    # All pixels reconstructed with the best algorithm (eta for 2 and 3, centroid otherwise)
    all_pixel_best_prefix = f"recon_zsuprec{zero_sup_threshold}_allpix_best"
    all_pixel_best = RESOLUTION_DIR / f"{file_prefix}_{all_pixel_best_prefix}.h5"
    if not all_pixel_best.exists():
        reconstruct(
            input_file=str(simulation_path),
            suffix=all_pixel_best_prefix,
            zero_sup_threshold=zero_sup_threshold,
            max_neighbors=6,
            pos_recon_algorithm="eta"
        )
    # Only one-pixel events with centroid
    one_pixel_centroid_prefix = f"recon_zsuprec{zero_sup_threshold}_1pix_centroid"
    one_pixel_centroid = RESOLUTION_DIR / f"{file_prefix}_{one_pixel_centroid_prefix}.h5"
    if not one_pixel_centroid.exists():
        reconstruct(
            input_file=str(simulation_path),
            suffix=one_pixel_centroid_prefix,
            zero_sup_threshold=zero_sup_threshold,
            max_neighbors=0,
            pos_recon_algorithm="centroid"
        )
    # Only two pixel events with centroid
    two_pixel_centroid_prefix = f"recon_zsuprec{zero_sup_threshold}_2pix_centroid"
    two_pixel_centroid = RESOLUTION_DIR / f"{file_prefix}_{two_pixel_centroid_prefix}.h5"
    if not two_pixel_centroid.exists():
        reconstruct(
            input_file=str(simulation_path),
            suffix=two_pixel_centroid_prefix,
            zero_sup_threshold=zero_sup_threshold,
            num_neighbors=1,
            pos_recon_algorithm="centroid"
        )
    # Only two-pixel events with eta
    two_pixel_eta_prefix = f"recon_zsuprec{zero_sup_threshold}_2pix_eta"
    two_pixel_eta = RESOLUTION_DIR / f"{file_prefix}_{two_pixel_eta_prefix}.h5"
    if not two_pixel_eta.exists():
        reconstruct(
            input_file=str(simulation_path),
            suffix=two_pixel_eta_prefix,
            zero_sup_threshold=zero_sup_threshold,
            num_neighbors=1,
            pos_recon_algorithm="eta"
        )
    # Only three-pixel events with centroid
    three_pixel_centroid_prefix = f"recon_zsuprec{zero_sup_threshold}_3pix_centroid"
    three_pixel_centroid = RESOLUTION_DIR / f"{file_prefix}_{three_pixel_centroid_prefix}.h5"
    if not three_pixel_centroid.exists():
        reconstruct(
            input_file=str(simulation_path),
            suffix=three_pixel_centroid_prefix,
            zero_sup_threshold=zero_sup_threshold,
            num_neighbors=2,
            pos_recon_algorithm="centroid"
        )
    # Only three-pixel events with eta
    three_pixel_eta_prefix = f"recon_zsuprec{zero_sup_threshold}_3pix_eta"
    three_pixel_eta = RESOLUTION_DIR / f"{file_prefix}_{three_pixel_eta_prefix}.h5"
    if not three_pixel_eta.exists():
        reconstruct(
            input_file=str(simulation_path),
            suffix=three_pixel_eta_prefix,
            zero_sup_threshold=zero_sup_threshold,
            num_neighbors=2,
            pos_recon_algorithm="eta"
        )
    # Define the hexagonal grid for pitch calculation and pixel center determination
    grid = HexagonalGrid()
    dr_binning = np.linspace(0., 0.6, 101)

    xlabel = "Distance residual [pitch normalized]"
    ylabel = "EEF"
    # 1 pixel events resolution
    dr_1pix, _ = distance(str(one_pixel_centroid), grid)
    hist_1pix = Histogram1d(dr_binning, xlabel=xlabel)
    hist_1pix.fill(dr_1pix)
    fig_1pix = plt.figure("1pix_residuals")
    hew = hist_1pix.ppf(0.5)
    plt.plot(dr_binning, hist_1pix.cdf(dr_binning), "-k", label=f"Centroid (HEW: {hew:.2f})")
    plt.hlines(0.5, dr_binning[0], dr_binning[-1], colors="0.4", linestyles="-.")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(dr_binning[0], dr_binning[-1])
    plt.ylim(0, 1)
    plt.legend()

    # 2 pixel events resolution with eta and centroid
    dr_2pix_eta, dr0_2pix_eta = distance(str(two_pixel_eta), grid)
    dr_2pix_centroid, dr0_2pix_centroid = distance(str(two_pixel_centroid), grid)
    hist_2pix_eta = Histogram1d(dr_binning, xlabel=xlabel)
    hist_2pix_eta.fill(dr_2pix_eta)
    hist_2pix_centroid = Histogram1d(dr_binning, xlabel=xlabel)
    hist_2pix_centroid.fill(dr_2pix_centroid)
    fig_2pix = plt.figure("2pix_residuals")
    hew_eta = hist_2pix_eta.ppf(0.5)
    hew_centroid = hist_2pix_centroid.ppf(0.5)
    plt.plot(dr_binning, hist_2pix_eta.cdf(dr_binning), "-k", label=f"$\\eta$ (HEW: {hew_eta:.2f})")
    plt.plot(dr_binning, hist_2pix_centroid.cdf(dr_binning), "--k", label=f"Centroid (HEW: {hew_centroid:.2f})")
    plt.hlines(0.5, dr_binning[0], dr_binning[-1], colors="0.4", linestyles="-.")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(dr_binning[0], dr_binning[-1])
    plt.ylim(0, 1)
    plt.legend()

    # 3 pixel events resolution with eta and centroid
    dr_3pix_eta, dr0_3pix_eta = distance(str(three_pixel_eta), grid)
    dr_3pix_centroid, dr0_3pix_centroid = distance(str(three_pixel_centroid), grid)
    hist_3pix_eta = Histogram1d(dr_binning, xlabel=xlabel)
    hist_3pix_eta.fill(dr_3pix_eta)
    hist_3pix_centroid = Histogram1d(dr_binning, xlabel=xlabel)
    hist_3pix_centroid.fill(dr_3pix_centroid)
    fig_3pix = plt.figure("3pix_residuals")
    hew_3pix_eta = hist_3pix_eta.ppf(0.5)
    hew_3pix_centroid = hist_3pix_centroid.ppf(0.5)
    plt.plot(dr_binning, hist_3pix_eta.cdf(dr_binning), "-k", label=f"$\\eta$ (HEW: {hew_3pix_eta:.2f})")
    plt.plot(dr_binning, hist_3pix_centroid.cdf(dr_binning), "--k", label=f"Centroid (HEW: {hew_3pix_centroid:.2f})")
    plt.hlines(0.5, dr_binning[0], dr_binning[-1], colors="0.4", linestyles="-.")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(dr_binning[0], dr_binning[-1])
    plt.ylim(0, 1)
    plt.legend()

    # All pixel events resolution with best algorithm and centroid
    dr_allpix_best, _ = distance(str(all_pixel_best), grid)
    dr_allpix_centroid, _ = distance(str(all_pixel_centroid), grid)
    hist_allpix_best = Histogram1d(dr_binning, xlabel=xlabel)
    hist_allpix_best.fill(dr_allpix_best)
    hist_allpix_centroid = Histogram1d(dr_binning, xlabel=xlabel)
    hist_allpix_centroid.fill(dr_allpix_centroid)
    fig_allpix = plt.figure("allpix_residuals")
    hew_allpix_best = hist_allpix_best.ppf(0.5)
    hew_allpix_centroid = hist_allpix_centroid.ppf(0.5)
    plt.plot(dr_binning, hist_allpix_best.cdf(dr_binning), "-k", label=f"Best (HEW: {hew_allpix_best:.2f})")
    plt.plot(dr_binning, hist_allpix_centroid.cdf(dr_binning), "--k", label=f"Centroid (HEW: {hew_allpix_centroid:.2f})")
    plt.hlines(0.5, dr_binning[0], dr_binning[-1], colors="0.4", linestyles="-.")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(dr_binning[0], dr_binning[-1])
    plt.ylim(0, 1)
    plt.legend()

    # Now we study the dependence of the resolution on the recon distance from the pixel center
    # For two pixel events eta and centroid
    hist_2d_2pix_eta = Histogram2d(
        dr_binning, dr_binning,
        xlabel="Reconstructed distance from pixel center [pitch normalized]",
        ylabel="Distance residual [pitch normalized]")
    hist_2d_2pix_eta.fill(dr0_2pix_eta, dr_2pix_eta)
    plt.figure("2pix_eta_residuals_vs_dr0")
    hist_2d_2pix_eta.plot()
    hist_2d_2pix_centroid = Histogram2d(
        dr_binning, dr_binning,
        xlabel="Reconstructed distance from pixel center [pitch normalized]",
        ylabel="Distance residual [pitch normalized]")
    hist_2d_2pix_centroid.fill(dr0_2pix_centroid, dr_2pix_centroid)
    plt.figure("2pix_centroid_residuals_vs_dr0")
    hist_2d_2pix_centroid.plot()
    # Take the slices and calculate the mean and stddev
    centers_2pix_eta, loc_2pix_eta, err_2pix_eta = _estimate_loc(hist_2d_2pix_eta)
    centers_2pix_centroid, loc_2pix_centroid, err_2pix_centroid = _estimate_loc(hist_2d_2pix_centroid)
    plt.figure("2pix_resolution_distance_dependency")
    plt.errorbar(
        centers_2pix_eta, loc_2pix_eta, yerr=err_2pix_eta,
        fmt=".r", label="2pix ETA")
    plt.errorbar(
        centers_2pix_centroid, loc_2pix_centroid, yerr=err_2pix_centroid,
        fmt=".b", label="2pix CENTROID")
    plt.xlabel("Reconstructed distance from pixel center [pitch normalized]")
    plt.ylabel("Mean distance residual [pitch normalized]")
    plt.legend()
    
    # For three pixel events eta and centroid
    hist_2d_3pix_eta = Histogram2d(
        dr_binning, dr_binning,
        xlabel="Reconstructed distance from pixel center [pitch normalized]",
        ylabel="Distance residual [pitch normalized]")
    hist_2d_3pix_eta.fill(dr0_3pix_eta, dr_3pix_eta)
    plt.figure("3pix_eta_residuals_vs_dr0")
    hist_2d_3pix_eta.plot()
    hist_2d_3pix_centroid = Histogram2d(
        dr_binning, dr_binning,
        xlabel="Reconstructed distance from pixel center [pitch normalized]",
        ylabel="Distance residual [pitch normalized]")
    hist_2d_3pix_centroid.fill(dr0_3pix_centroid, dr_3pix_centroid)
    plt.figure("3pix_centroid_residuals_vs_dr0")
    hist_2d_3pix_centroid.plot()
    # Take the slices and calculate the mean and stddev
    centers_3pix_eta, loc_3pix_eta, err_3pix_eta = _estimate_loc(hist_2d_3pix_eta)
    centers_3pix_centroid, loc_3pix_centroid, err_3pix_centroid = _estimate_loc(hist_2d_3pix_centroid)
    plt.figure("3pix_resolution_distance_dependency")
    plt.errorbar(
        centers_3pix_eta, loc_3pix_eta, yerr=err_3pix_eta,
        fmt=".r", label="3pix ETA")
    plt.errorbar(
        centers_3pix_centroid, loc_3pix_centroid, yerr=err_3pix_centroid,
        fmt=".b", label="3pix CENTROID")
    plt.xlabel("Reconstructed distance from pixel center [pitch normalized]")
    plt.ylabel("Mean distance residual [pitch normalized]")
    plt.legend()

    if kwargs["save"]:
        fig_format = "png"
        fig_1pix.savefig(FIGURES_DIR / f"1pix_resolution_{enc}enc_{zero_sup_threshold}zsup.{fig_format}", format=fig_format)
        fig_2pix.savefig(FIGURES_DIR / f"2pix_resolution_{enc}enc_{zero_sup_threshold}zsup.{fig_format}", format=fig_format)
        fig_3pix.savefig(FIGURES_DIR / f"3pix_resolution_{enc}enc_{zero_sup_threshold}zsup.{fig_format}", format=fig_format)
        fig_allpix.savefig(FIGURES_DIR / f"allpix_resolution_{enc}enc_{zero_sup_threshold}zsup.{fig_format}", format=fig_format)


if __name__ == "__main__":
    resolution(**vars(HXETA_ARGPARSER.parse_args()))
    plt.show()
