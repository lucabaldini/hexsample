# Copyright (C) 2023--2025 the hexsample team.
#
# For the license terms see the file LICENSE, distributed along with this
# software.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import argparse
from pathlib import Path

import numpy as np
from aptapy.plotting import plt

from hexsample.fileio import ReconInputFile
from hexsample.hexagon import HexagonalLayout
from hexsample.pipeline import reconstruct, simulate
from hexsample.resolution import eef, eef_size_scan, eew, resolution_spatial_dependence

__description__ = ""

# Parser object.
HXETA_ARGPARSER = argparse.ArgumentParser(description=__description__)
HXETA_ARGPARSER.add_argument("enc", type=int,
                             help="equivalent noise charge in electrons")
HXETA_ARGPARSER.add_argument("zero_sup_threshold", type=int,
                             help="zero suppression threshold in electrons")
HXETA_ARGPARSER.add_argument("--save", action="store_true",
                             help="save the figures")

RESOLUTION_DIR = Path.home() / "hexsampledata" / "resolution"
if not RESOLUTION_DIR.exists():
    RESOLUTION_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = Path.home() / "hexsample_figures" / "resolution"
if not FIGURES_DIR.exists():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def resolution(**kwargs):
    """Run the resolution analysis. This analysis consists of multiple steps to study different
    aspects of the resolution.
    
    First the Encircled Energy Function (EEF) is created for different cluster sizes and
    reconstruction algorithms for a given ENC and zero suppression threshold.

    Then the spatial dependence of the resolution is studied by plotting the HEW as a function of
    the reconstructed distance from the true pixel center.
    
    Finally, for the given ENC, the EEF is plotted for different zero suppression thresholds and
    for both the centroid and eta algorithms.
    """
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
    sp_dep_fig = plt.figure("resolution_spatial_dependence")
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
    zero_sup_ratios = np.linspace(0, 3, 13)
    recon_kwargs = dict(input_file=str(simulation_path),
                        max_neighbors=6)
    eef_zsup_centroid_fig = plt.figure(f"eef_vs_zsup_centroid_enc{enc}")
    print("Reconstructing files for centroid algorithm...")
    for zero_sup_ratio in zero_sup_ratios:
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
    plt.xlabel(xlabel = r"$r/p$")
    plt.ylabel("Encircled Energy Fraction")
    plt.xlim(x[0], x[-1])
    plt.ylim(0, 1)
    plt.legend()

    eef_zsup_best_fig = plt.figure(f"eef_vs_zsup_best_enc{enc}")
    print("Reconstructing files for eta algorithm...")
    eew_list = []
    for zero_sup_ratio in zero_sup_ratios:
        zsup = int(zero_sup_ratio * enc)
        suffix = f"recon_zsuprec{zsup}_best"
        file_path = RESOLUTION_DIR / f"{file_prefix}_{suffix}.h5"
        if not file_path.exists():
            reconstruct(suffix=suffix, pos_recon_algorithm="eta", zero_sup_threshold=zsup,
                        **recon_kwargs)
        # Open file and plot EEF
        recon_file = ReconInputFile(str(file_path))
        plt.plot(x, eef(x, recon_file, max_neighbors=6), label=f"zsup/enc {zero_sup_ratio}")
        eew_list.append(eew(recon_file, quantile=0.9, max_neighbors=6))
        recon_file.close()
    plt.xlabel(xlabel = r"$r/p$")
    plt.ylabel("Encircled Energy Fraction")
    plt.xlim(x[0], x[-1])
    plt.ylim(0, 1)
    plt.legend()

    plt.figure("encircled_energy_width_@0.9vs_zsup")
    plt.plot(zero_sup_ratios, eew_list, ".k")
    plt.xlabel("Zero suppression threshold / ENC")
    plt.ylabel("Encircled Energy Width @ 0.9")

    # Save figures, if requested
    if kwargs["save"]:
        fig_format = "png"
        zsup_th = zero_sup_threshold
        centroid_fig.savefig(FIGURES_DIR / f"centroid_eef_{enc}enc_{zsup_th}zsup.{fig_format}",
                             format=fig_format)
        best_fig.savefig(FIGURES_DIR / f"best_eef_{enc}enc_{zsup_th}zsup.{fig_format}",
                         format=fig_format)
        sp_dep_fig.savefig(FIGURES_DIR / f"res_spatial_depend_{enc}enc_{zsup_th}zsup.{fig_format}",
                           format=fig_format)
        eef_zsup_centroid_fig.savefig(FIGURES_DIR / f"eef_vs_zsup_centroid_{enc}enc.{fig_format}",
                                      format=fig_format)
        eef_zsup_best_fig.savefig(FIGURES_DIR / f"eef_vs_zsup_best_{enc}enc.{fig_format}",
                                  format=fig_format)



if __name__ == "__main__":
    resolution(**vars(HXETA_ARGPARSER.parse_args()))
    plt.show()
