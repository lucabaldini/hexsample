import argparse
from pathlib import Path

import numpy as np
from aptapy.hist import Histogram2d
from aptapy.plotting import plt

from hexsample.fileio import ReconInputFile
from hexsample.hexagon import HexagonalLayout
from hexsample.pipeline import reconstruct, simulate
from hexsample.resolution import eef, eef_size_scan, resolution_spatial_dependence

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
                num_events=100000,
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
    recon_kwargs = dict(input_file=str(simulation_path),
                        zero_sup_threshold=zero_sup_threshold,
                        max_neighbors=6)
    # Reconstruct the simulated file with different algorithms, first with centroid
    centroid_prefix = f"recon_zsuprec{zero_sup_threshold}_centroid"
    centroid_path = RESOLUTION_DIR / f"{file_prefix}_{centroid_prefix}.h5"
    if not centroid_path.exists():
        reconstruct(suffix=centroid_prefix, pos_recon_algorithm="centroid", **recon_kwargs)
    # Reconstruct with the best algorithm (eta for 2 and 3, centroid otherwise)
    best_prefix = f"recon_zsuprec{zero_sup_threshold}_best"
    best_path = RESOLUTION_DIR / f"{file_prefix}_{best_prefix}.h5"
    if not best_path.exists():
        reconstruct(suffix=best_prefix, pos_recon_algorithm="eta",**recon_kwargs)

    # Open the recon files
    centroid_recon_file = ReconInputFile(str(centroid_path))
    best_recon_file = ReconInputFile(str(best_path))
    # Define the plotting range
    x = np.linspace(0, 0.6, 101)
    # Plot the centroid eef for all cluster sizes
    centroid_fig = plt.figure(f"centroid_eef_{enc}enc_{zero_sup_threshold}zsup")
    eef_size_scan(x, centroid_recon_file)
    # Plot the eta + centroid eef for all cluster sizes
    best_fig = plt.figure(f"best_eef_{enc}enc_{zero_sup_threshold}zsup")
    eef_size_scan(x, best_recon_file)

    # Plot the spatial dependence of the resolution for both algorithms (all events)
    spatial_dependence_fig = plt.figure("resolution_spatial_dependence")
    plt.plot(*resolution_spatial_dependence(best_recon_file, max_neighbors=6),
             ".k", label=r"$\eta$ + centroid")
    plt.plot(*resolution_spatial_dependence(centroid_recon_file, max_neighbors=6),
             "vk", label="centroid", markersize=4.)
    plt.xlabel(r"$r_0 / p$")
    plt.ylabel("Half Energy Width")
    plt.legend()

    # Close the recon files
    centroid_recon_file.close()
    best_recon_file.close()

    # Study the resolution as a function of zero sup threshold for both algorithms
    zero_sup_ratios = np.linspace(0, 3, 4)
    recon_kwargs = dict(input_file=str(simulation_path),
                        max_neighbors=6)
    eef_zsup_centroid_fig = plt.figure(f"eef_vs_zsup_centroid_enc{enc}")
    print("Reconstructing files for centroid algorithm...")
    for i, zero_sup_ratio in enumerate(zero_sup_ratios):
        zsup = int(zero_sup_ratio * enc)
        suffix = f"recon_zsuprec{zsup}_centroid"
        file_path = RESOLUTION_DIR / f"{file_prefix}_{suffix}.h5"
        if not file_path.exists():
            reconstruct(suffix=suffix, pos_recon_algorithm="centroid", zero_sup_threshold=zsup,
                        **recon_kwargs)
        # Open file and plot EEF
        recon_file = ReconInputFile(str(file_path))
        plt.plot(x, eef(x, recon_file, max_neighbors=6), label=f"zsup/enc {zero_sup_ratio}")
        recon_file.close()

    eef_zsup_best_fig = plt.figure(f"eef_vs_zsup_best_enc{enc}")
    print("Reconstructing files for eta algorithm...")
    for i, zero_sup_ratio in enumerate(zero_sup_ratios):
        zsup = int(zero_sup_ratio * enc)
        suffix = f"recon_zsuprec{zsup}_best"
        file_path = RESOLUTION_DIR / f"{file_prefix}_{suffix}.h5"
        if not file_path.exists():
            reconstruct(suffix=suffix, pos_recon_algorithm="eta", zero_sup_threshold=zsup,
                        **recon_kwargs)
        # Open file and plot EEF
        recon_file = ReconInputFile(str(file_path))
        plt.plot(x, eef(x, recon_file, max_neighbors=6), label=f"zsup/enc {zero_sup_ratio}")
        recon_file.close()

    plt.xlabel(xlabel = r"$r/p$")
    plt.ylabel("Encircled Energy Fraction")
    plt.xlim(x[0], x[-1])
    plt.ylim(0, 1)
    plt.legend()

    if kwargs["save"]:
        fig_format = "png"
        centroid_fig.savefig(FIGURES_DIR / f"centroid_eef_{enc}enc_{zero_sup_threshold}zsup.{fig_format}", format=fig_format)
        best_fig.savefig(FIGURES_DIR / f"best_eef_{enc}enc_{zero_sup_threshold}zsup.{fig_format}", format=fig_format)
        spatial_dependence_fig.savefig(FIGURES_DIR / f"resolution_spatial_dependence_{enc}enc_{zero_sup_threshold}zsup.{fig_format}", format=fig_format)
        eef_zsup_centroid_fig.savefig(FIGURES_DIR / f"eef_vs_zsup_centroid_{enc}enc.{fig_format}", format=fig_format)
        eef_zsup_best_fig.savefig(FIGURES_DIR / f"eef_vs_zsup_best_{enc}enc.{fig_format}", format=fig_format)



if __name__ == "__main__":
    resolution(**vars(HXETA_ARGPARSER.parse_args()))
    plt.show()
