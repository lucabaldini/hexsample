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

from dataclasses import dataclass
from typing import Tuple

from matplotlib.pyplot import hist
import numpy as np
from aptapy.hist import Histogram1d, Histogram2d
from aptapy.plotting import plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import tukey
from skimage import feature, transform

from .fileio import ReconInputFile
from .hexagon import HexagonalGrid


@dataclass
class SlitsAligner:

    """Align the slits of a Huttner test pattern to the detector axes by estimating the tilt angle
    of the slits with respect to the detector axes. To find the tilt angle, first the Canny edge
    detection is applied to the reconstructed positions to find the edges of the slits. Then the
    Hough transform is applied to the detected edges to find the angle.

    Attributes
    ----------
    bin_size : float
        The bin size to be used for the 2D histogram of the reconstructed positions. This
        histogram is needed to apply the Canny edge detection. The bin size should be smaller than
        the pixel size, but not too small to avoid having too much pixels in the image.
    sigma : float
        The standard deviation of the Gaussian filter to be applied to the image during the Canny
        edge detection. If its value is too small, the pixel edges are detected instead of the slit
        edges. Otherwise, a too large value can lead to no edge being detected.
    """
    bin_size: float = 0.001
    sigma: float = 10.

    def _detect_edges(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Detect the slit edges of the Huttner test pattern to align the slit to the detector
        axes. The method is based on the Canny edge detection, which allows to detect the edges
        in images with noise.

        Arguments
        ---------
        x : np.ndarray
            The x coordinates of the reconstructed positions.
        y : np.ndarray
            The y coordinates of the reconstructed positions.
        
        Returns
        -------
        edges : np.ndarray
            The image with the detected edges.
        """
        # Before applying the Canny edge detection, we need an image. We create a 2D histogram
        xedges = np.arange(x.min(), x.max(), step=self.bin_size)
        yedges = np.arange(y.min(), y.max(), step=self.bin_size)
        image, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
        # Apply Canny edge detection to the binned image.
        return feature.canny(image.T, sigma=self.sigma)

    def _estimate_angle(self, edges: np.ndarray) -> float:
        """Estimate the tilt angle of the slits with respect to the detector axes by applyting the
        Hough transform to the slits edge image calculated with the Canny edge detection.

        Arguments
        ---------
        edges : np.ndarray
            The image with the detected edges.
        
        Returns
        -------
        theta : float
            The estimated tilt angle of the slits with respect to the detector x-axis in
            counterclockwise direction.
        """
        # Calculate the Hough transform of the detected edges. We test a small range of angles
        # as we expect the slits to be tilted only by a few degrees with respect to the detector
        # x-axis. These angles are reffered to the normal to the edges.
        test_angles = np.deg2rad(np.linspace(80, 100, 1000))
        hspace, angles, distances = transform.hough_line(edges, theta=test_angles)
        # Extract the peaks from the Hough transform.
        _, peaks_angles, _ = transform.hough_line_peaks(hspace, angles, distances, num_peaks=10)
        # Calculate the tilt angle as pi/2 minus the mean angle of the detected peaks.
        # The pi/2 is needed because the Hough transform returns the angle of the normal to the
        # line.
        return np.pi / 2  - np.mean(peaks_angles)

    def align(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate the reconstructed positions to align the slits to the detector x-axis.

        To calculate the tilt angle, this method first bins the data in an image and then finds the
        edges of the slits by applying Canny edge detection. Finally the angle is computed by
        applying the Hough transform to the detected edges.

        Arguments
        ---------
        x : np.ndarray
            The x coordinates of the reconstructed positions.
        y : np.ndarray
            The y coordinates of the reconstructed positions.
        
        Returns
        -------
        x : np.ndarray
            The x coordinates of the reconstructed positions aligned to the detector x-axis.
        y : np.ndarray
            The y coordinates of the reconstructed positions aligned to the detector x-axis.
        """
        # Detect the edges of the slits
        edges = self._detect_edges(x, y)
        # Estimate the tilt angle
        theta = self._estimate_angle(edges)
        # Rotate the reconstruction positions by the estimated angle
        x_aligned = x * np.cos(theta) - y * np.sin(theta)
        y_aligned = x * np.sin(theta) + y * np.cos(theta)
        return x_aligned, y_aligned


@dataclass
class SlantedEdgeResolution:

    """Calculate the resolution of the detector by applying the slanted edge method to the
    reconstructed positions. This method consists of finding the edge spread function (ESF) of
    a test pattern with a slanted edge. The line spread function (LSF) is calculated as the
    derivative of the ESF. Finally, the modulation transfer function (MTF) is calculated as the
    Fourier transform of the LSF.

    Attributes
    ----------
    x : np.ndarray
        The reconstructed coordinates of the events along the direction perpendicular to the edge.
    bin_size: float
        The bin size to be used for the ESF calculation.
    sigma: float
        The standard deviation of the Gaussian filter to smooth the ESF. It is expressed in number
        of bins. Consider that the smoothing acts as a low-pass filter, so if it is too large it
        can lead to an underestimation of the resolution.
    """

    x : np.ndarray
    bin_size: float = 0.0002
    sigma: float = 2

    def _esf(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the edge spread function (ESF) of the slanted edge. A padding is added to the
        left and right of the histogram range to avoid problems at the edges of the domain. The
        data are binned and then smoothed with a Gaussian filter to reduce the noise and geometric
        effects on the plateu.
        
        Note that the reconstructed positions are expected to be aligned to the axes of the detector.

        Returns
        -------
        esf : np.ndarray
            The edge spread function (ESF) of the slanted edge.
        edges : np.ndarray
            The edges of the bins used for the ESF calculation.
        """
        # Add some padding to the histogram range to avoid problems at the edges of the histogram
        # when smoothing the ESF with a Gaussian filter. 
        padding = self.bin_size * 50
        x_min, x_max = self.x.min() - padding, self.x.max() + padding
        # When using a large slit to calculate the ESF, we cut the histogram at the center of the
        # slit to avoid analyzing two edges at the same time.
        x_center = (x_min + x_max) / 2
        edges = np.arange(x_min, x_center, step=self.bin_size)
        # Prepare the data for the smoothing with a Gaussian filter.
        counts, _ = np.histogram(self.x[self.x < x_center], bins=edges)
        smoothet_esf = gaussian_filter1d(counts, sigma=self.sigma)
        return smoothet_esf, edges
    
    @property
    def esf(self) -> Histogram1d:
        """Calculate the edge spread function (ESF) of the slanted edge as a Histogram1d object.

        Returns
        -------
        esf : Histogram1d
            The edge spread function (ESF) of the slanted edge.
        """
        # Calculate the smoothed ESF from the reconstructed positions. 
        esf_values, edges = self._esf()
        # Create the histogram and set the content to the ESF values.
        hist = Histogram1d(edges)
        hist.set_content(esf_values)
        return hist

    def _lsf(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the line spread function (LSF) from the ESF by calculating its derivative.

        Returns
        -------
        lsf : np.ndarray 
            The line spread function (LSF) of the slanted edge.
        edges : np.ndarray
            The edges of the bins used for the LSF calculation.
        """
        # Calculate the ESF
        esf, esf_edges = self._esf()
        # Calculate the derivative of the ESF and the corresponding bin edges.
        lsf = np.diff(esf) / self.bin_size
        edges = esf_edges[1:]
        # Cut the data after the LSF reaches zero to avoid the noise from the plateu. We should
        # think of a more robust way dto do this, such as a windowing function.
        centers = (edges[:-1] + edges[1:]) / 2
        zero_crossing = centers[lsf < 0][0]
        lsf[centers >= zero_crossing] = 0
        # Normalize the LSF histogram, so that the MTF maximum is 1.
        lsf /= np.sum(lsf) * self.bin_size
        return lsf, edges

    @property
    def lsf(self) -> Histogram1d:
        """Calculate the line spread function (LSF) of the slanted edge as a Histogram1d object.

        Returns
        -------
        lsf : Histogram1d
            The line spread function (LSF) of the slanted edge.
        """
        # Calculate the LSF from the ESF.
        lsf_values, edges = self._lsf()
        # Create the histogram and set the content to the LSF values.
        hist = Histogram1d(edges)
        hist.set_content(lsf_values)
        return hist
    
    def mtf(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the modulation transfer function (MTF) of the slanted edge by calculating the
        Fourier transform of the LSF.

        Returns
        -------
        mtf : np.ndarray
            The modulation transfer function (MTF) of the slanted edge.
        frequency : np.ndarray
            The spatial frequencies corresponding to the MTF values.
        """
        # Calculate the LSF
        lsf, _ = self._lsf()
        # Calculate the Fourier transform of the LSF and take its absolute value to get the MTF.
        # Also calculate the corresponding spatial frequencies.
        mtf_fft = np.abs(np.fft.fft(lsf))
        freq_fft = np.fft.fftfreq(len(lsf), d=self.bin_size)
        # Consider only the first half of the MTF, which corresponds to the positive spatial
        # frequencies. The MTF is symmetric for real signals, so we can ignore the second half.
        mtf = mtf_fft[:len(lsf) // 2]
        freqs = freq_fft[:len(lsf) // 2]
        # Normalize the MTF so that its value at zero frequency is 1.
        mtf /= mtf[0]
        return mtf, freqs


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
    # Calculate the reconstructed distance from the pixel center
    dr0 = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return dr0


def hist_distance_residuals(input_file: ReconInputFile, num_neighbors: int = 0,
                            max_neighbors: int = -1) -> Histogram1d:
    """Create the histogram of distance residuals for the given number of neighbors.

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
    # spans all the pitch.
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
    hist = hist_distance_residuals(input_file, num_neighbors, max_neighbors)
    return hist.cdf(x)


def eew(input_file: ReconInputFile, quantile: float, num_neighbors: int = 0,
        max_neighbors: int = -1) -> float:
    """Calculate the Encircled Energy Width (EEW) for a given quantile and cluster size.

    Arguments
    ---------
    input_file : ReconInputFile
        The input file to analyze.
    quantile : float
        The quantile to be used for the EEW calculation.
    num_neighbors : int, optional
        The number of neighbors to be considered. Default is 0.
    max_neighbors : int, optional
        The maximum number of neighbors to be considered. If max_neighbors is specified, it has
        priority over num_neighbors. Default is -1 (not used).

    Returns
    -------
    eew : float
        The Encircled Energy Width (EEW) evaluated at a given quantile, expressed in pitch
        normalized units.
    """
    if not 0 <= quantile <= 1:
        raise ValueError(f"Quantile must be between 0 and 1, got {quantile}.")
    hist = hist_distance_residuals(input_file, num_neighbors, max_neighbors)
    return hist.ppf(quantile)


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
    return eew(input_file, quantile=0.5,
               num_neighbors=num_neighbors,
               max_neighbors=max_neighbors)


def eef_size_scan(x: np.ndarray, input_file: ReconInputFile) -> None:
    """Plot the Encircled Energy Function for one, two, three pixel events and for all the
    reconstructed events on the same figure.
    
    Arguments
    ---------
    x : np.ndarray
        The pitch normalized distance values where the EEF will be evaluated.
    input_file : ReconInputFile
        The input file to analyze.
    """
    xlabel = r"$r/p$"
    ylabel = "Encircled Energy Fraction"
    color = "black"
    # Plot the EEFs for the different cluster sizes
    plt.plot(x, eef(x, input_file, 0),
             label=f"1 pix (EEF@86.5%={eew(input_file, quantile=0.865, num_neighbors=0):.2f})",
             linestyle=":", color=color)
    plt.plot(x, eef(x, input_file, 1),
             label=f"2 pix (EEF@86.5%={eew(input_file, quantile=0.865, num_neighbors=1):.2f})",
             linestyle="-", color=color)
    plt.plot(x, eef(x, input_file, 2),
             label=f"3 pix (EEF@86.5%={eew(input_file, quantile=0.865, num_neighbors=2):.2f})",
             linestyle="--", color=color)
    plt.plot(x, eef(x, input_file, max_neighbors=6),
             label=f"All events (EEF@86.5%={eew(input_file, quantile=0.865, max_neighbors=6):.2f})",
             linestyle="-.", color=color)
    # Plot the line corresponding to 50% of the energy
    plt.hlines(0.5, x[0], x[-1], colors="0.4", linestyles="--")
    # Finalize the plot
    plt.xlim(x[0], x[-1])
    plt.ylim(0, 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()


def resolution_spatial_dependence(input_file: ReconInputFile, quantile: float,
                                  num_neighbors: int = 1, max_neighbors: int = -1
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the spatial dependence of the resolution. This is estimated by calculating the
    Encircled Energy Width (EEW) as a function of the reconstructed distance from the true pixel
    center.

    Arguments
    ---------
    input_file : ReconInputFile
        The input file to analyze.
    num_neighbors : int, optional
        The number of neighbors to be considered. Default is 1.
    max_neighbors : int, optional
        The maximum number of neighbors to be considered. If max_neighbors is specified, it has
        priority over num_neighbors. Default is -1 (not used).
    
    Returns
    -------
    bin_centers : np.ndarray
        The bin centers of the reconstructed distance from the pixel center.
    eews : np.ndarray
        The Encircled Energy Width (EW) calculated for each bin of the reconstructed distance from
        the pixel center.
    """
    # Select the cluster sizes we want and create the mask
    size = input_file.column("cluster_size")
    mask = size <= max_neighbors + 1 if max_neighbors >= 0 else size == num_neighbors + 1
    # Calculate the pitch normalized distance from pixel center and distance residuals
    pitch = input_file.digi_header["pitch"]
    dr0 = dist_from_pixel_center(input_file)[mask] / pitch
    dr = dist_residual(input_file)[mask] / pitch
    # Create the 2D histogram
    xedges = np.linspace(0., 1., 101)
    if dr.size == 0:
        # If there are no events, raise an error
        raise ValueError("No events found for the given cluster size.")
    yedges = np.linspace(min(dr), max(dr), 101)
    hist = Histogram2d(xedges, yedges)
    hist.fill(dr0, dr)
    # Calculate the Encircled Energy Width (EW) for each slice of the histogram
    bin_centers = hist.bin_centers()
    eews = np.zeros(len(bin_centers))
    for i in range(len(bin_centers)):
        xslice = hist.slice1d(i)
        if xslice.content.sum() == 0:
            eews[i] = np.nan
        else:
            eews[i] = xslice.ppf(quantile)
    return bin_centers, eews
