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

"""Resolution analysis facilities.
"""

import numpy as np
from aptapy.hist import Histogram1d
from aptapy.plotting import plt

from .fileio import ReconInputFile
from .hexagon import HexagonalGrid


def dist_residual(input_file: ReconInputFile) -> np.ndarray:
    """Calculate the distance residuals between reconstructed and Monte Carlo true positions.

    Arguments
    ---------
    input_file : ReconInputFile
        The input file to analyze.

    Returns
    -------
    dr : np.ndarray
        The distance residuals.
    """
    # Access the Monte Carlo true positions
    x_mc = input_file.mc_column("absx")
    y_mc = input_file.mc_column("absy")
    # Access the reconstructed positions
    x = input_file.column("posx")
    y = input_file.column("posy")
    # Calculate the distance residuals
    dr = np.sqrt((x - x_mc) ** 2 + (y - y_mc) ** 2)
    return dr


def dist_from_pixel_center(input_file: ReconInputFile) -> np.ndarray:
    """Calculate the reconstructed distance from the pixel center.
    
    Arguments
    ---------
    input_file : ReconInputFile
        The input file to analyze.
    
    Returns
    -------
    dr0 : np.ndarray
        The reconstructed distance from the pixel center.
    """
    # Create the hexagonal grid to get pixel centers
    layout = input_file.digi_header["layout"]
    num_cols = input_file.digi_header["num_cols"]
    num_rows = input_file.digi_header["num_rows"]
    grid = HexagonalGrid(layout=layout, num_cols=num_cols, num_rows=num_rows)
    # Access Monte Carlo true positions
    x_mc = input_file.mc_column("absx")
    y_mc = input_file.mc_column("absy")
    # Calculate Monte Carlo true pixel centers
    x0, y0 = grid.pixel_to_world(*grid.world_to_pixel(x_mc, y_mc))
    # Access the reconstructed positions
    x = input_file.column("posx")
    y = input_file.column("posy")
    # Calulate the reconstructed distance from the pixel center
    dr0 = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return dr0


def hist_distance_res(input_file: ReconInputFile, num_neighbors: int = 0,
                      max_neighbors: int = -1) -> Histogram1d:
    """Create the histogram of distance residuals for the given numqber of neighbors.

    Arguments
    ---------
    input_file : ReconInputFile
        The input file to analyze.
    num_neighbors : int, optional
        The number of neighbors to be considered. Default is 0.
    max_neighbors : int, optional
        The maximum number of neighbors to be considered. If max_neighbors is specified, it has
        priority over num_neighbors. Default is -1 (not used).

    Returns
    -------
    hist : Histogram1d
        The histogram of pitch normalized distance residuals.
    """
    # Select the cluster sizes we want and create the mask
    size = input_file.column("cluster_size")
    mask = size <= max_neighbors + 1 if max_neighbors >= 0 else size == num_neighbors + 1
    pitch = input_file.digi_header["pitch"]
    # Calculate the distance residuals normalized to pitch
    dr = dist_residual(input_file) / pitch
    # Create the histogram to calculate the EEF. The binning is taken in a way that
    # it covers the full range of dr.
    dr_binning = np.linspace(0., 1., 101)
    hist = Histogram1d(dr_binning)
    hist.fill(dr[mask])
    return hist


def eef(x: np.ndarray, input_file: ReconInputFile, num_neighbors: int = 0,
        max_neighbors: int = -1) -> np.ndarray:
    """Calculate the Encircled Energy Function (EEF) for a given cluster size.

    Arguments
    ---------
    x : np.ndarray
        The normalized distance values where the EEF will be evaluated.
    input_file : ReconInputFile
        The input file to analyze.
    num_neighbors : int, optional
        The number of neighbors to be considered. Default is 0.
    max_neighbors : int, optional
        The maximum number of neighbors to be considered. If max_neighbors is specified, it has
        priority over num_neighbors. Default is -1 (not used).

    Returns
    -------
    eef : np.ndarray
        The Encircled Energy Function evaluated at the given normalized distance values.
    """
    hist = hist_distance_res(input_file, num_neighbors, max_neighbors)
    return hist.cdf(x)


def hew(input_file: ReconInputFile, num_neighbors: int = 0,
        max_neighbors: int = -1) -> float:
    """Calculate the Half Energy Width (HEW) for a given cluster size.

    Arguments
    ---------
    input_file : ReconInputFile
        The input file to analyze.
    num_neighbors : int, optional
        The number of neighbors to be considered. Default is 0.
    max_neighbors : int, optional
        The maximum number of neighbors to be considered. If max_neighbors is specified, it has
        priority over num_neighbors. Default is -1 (not used).

    Returns
    -------
    hew : float
        The Half Energy Width (HEW) in pitch normalized units.
    """
    hist = hist_distance_res(input_file, num_neighbors, max_neighbors)
    return hist.ppf(0.5)


def eef_size_scan(x: np.ndarray, input_file: ReconInputFile) -> None:
    """Plot the EEF for one, two, three pixel events and for all the reconstructed events on
    the same figure.
    
    Arguments
    ---------
    x : np.ndarray
        The pitch normalized distance values where the EEF will be evaluated.
    input_file : ReconInputFile
        The input file to analyze.
    """
    xlabel = "r/p"
    ylabel = "Encircled Energy Fraction"
    # Plot the EEFs for the different cluster sizes
    plt.plot(x, eef(x, input_file, 0),
             label=f"1 pix (HEW={hew(input_file, 0):.2f})",
             linestyle=":", color="black")
    plt.plot(x, eef(x, input_file, 1),
             label=f"2 pix (HEW={hew(input_file, 1):.2f})",
             linestyle="-", color="black")
    plt.plot(x, eef(x, input_file, 2),
             label=f"3 pix (HEW={hew(input_file, 2):.2f})",
             linestyle="--", color="black")
    plt.plot(x, eef(x, input_file, max_neighbors=6),
             label=f"All pix (HEW={hew(input_file, max_neighbors=6):.2f})",
             linestyle="-.", color="black")
    # Finalize the plot
    plt.xlim(x[0], x[-1])
    plt.ylim(0, 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
