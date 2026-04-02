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

"""Clustering facilities.
"""

from dataclasses import dataclass
from typing import Tuple

import iminuit
import numpy as np
from aptapy.models import Probit

from .digi import DigiEventCircular, DigiEventRectangular
from .likelihood import nll_numba, nll_grad_numba, nll_numba_profiled, nll_grad_numba_profiled
from .readout import HexagonalReadoutBase


@dataclass
class Cluster:

    """Small container class describing a cluster.
    """

    # pylint: disable = invalid-name

    x: np.ndarray
    y: np.ndarray
    col: np.ndarray
    row: np.ndarray
    pha: np.ndarray
    pos_recon_algorithm: str
    recon_pars: dict
    _errx_low: float = 0.
    _errx_high: float = 0.
    _erry_low: float = 0.
    _erry_high: float = 0.

    def __post_init__(self) -> None:
        """Small cross check on the dimensions of the arrays passed in the constructor.
        """
        if not self.x.shape == self.y.shape == self.pha.shape:
            raise RuntimeError(f"Inconsistent arrays: x = {self.x}, y = {self.y}, pha = {self.pha}")

    def size(self) -> int:
        """Return the size of the cluster.
        """
        return self.x.size

    def pulse_height(self) -> float:
        """Return the total pulse height of the cluster.
        """
        return self.pha.sum()

    def centroid(self) -> Tuple[float, float]:
        """Return the cluster centroid.
        """
        return np.average(self.x, weights=self.pha), np.average(self.y, weights=self.pha)

    def calculate_eta(self) -> np.ndarray:
        """Return the eta values of the pixels in the cluster.
        """
        return np.array([_pha / self.pulse_height() for _pha in self.pha[1:]])

    def versors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the versors u and v for the cluster. Their definitions depend on the cluster size.
        For 2-pixel clusters u is the versor that points from the center of the pixel with the
        highest pha to the center of the other one, while v is the versor perpendicular to u in
        counterclockwise direction. For 3-pixel clusters u points from the center
        of the pixel with the highest pha to the midpoint of the line that connects the centers of
        the other two pixels, and v is perpendicular to u and points towards the second most
        charged pixel.

        Returns
        -------
        u : np.ndarray
            The u versor.
        v : np.ndarray
            The v versor.
        """
        if self.x.shape[0] == 2:
            u = np.array([self.x[1] - self.x[0], self.y[1] - self.y[0]])
            v = np.array([-u[1], u[0]])
        elif self.x.shape[0] == 3:
            u = np.array([self.x[1] + self.x[2] - 2 * self.x[0],
                          self.y[1] + self.y[2] - 2 * self.y[0]])
            v = np.array([-u[1], u[0]])
            if (self.x[1] - self.x[0]) * v[0] + (self.y[1] - self.y[0]) * v[1] < 0:
                v = -v
        else:
            raise RuntimeError("Cluster must contain 2 or 3 pixels to calculate versors")
        # It can happen that the versor is [0, 0] for events with strange geometries.
        # In that case we avoid NaN by setting the versor to [0, 0].
        with np.errstate(invalid="ignore"):
            norm = np.sqrt(np.sum(u**2))
            if norm > 0:
                u = u / norm
                v = v / norm
            else:
                u = np.zeros(2)
                v = np.zeros(2)
        return u, v

    def eta(self, s2: float, p2: float, mu3_r: float, s3_r: float, p3_r: float, s3_t: float,
            pitch: float) -> Tuple[float, float]:
        """Return the cluster reconstructed position using the eta function calibrated for 2
        and 3 pixel clusters. If cluster size is not 2 or 3, reconstruct the position with the
        centroid.

        Arguments
        ---------
        s2 : float
            Probit function sigma parameter for two pixel events.
        p2 : float
            Transition value from linear (0 to pivot) to probit (> pivot) for two pixel events.
        mu3_r : float
            Probit function offset parameter for three pixel events radial position component.
        s3_r : float
            Probit function sigma parameter for three pixel events radial position component.
        p3_r : float
            Transition value from linear (0 to pivot) to probit (> pivot) for three pixel events
            radial position component.
        s3_t : float
            Probit function sigma parameter for three pixel events angular position component.
        pitch : float
            The pitch of the pixels.
        """
        # Return the centroid position if it's not possible to use the eta function
        if self.size() not in (2, 3):
            return self.centroid()
        # Calculate versors and eta.
        u, v = self.versors()
        _eta = self.calculate_eta()

        if self.size() == 2:
            # For 2-pixel events we estimate the position along the line that connects the
            # two pixels using the probit function.
            if _eta[0] > p2 or p2 <= 0.:
                r = Probit().evaluate(_eta[0], 0.5, s2)
            else:
                y_pivot = Probit().evaluate(p2, 0.5, s2)
                r = y_pivot / p2 * _eta[0]
            x_recon = self.x[0] + r * pitch * u[0]
            y_recon = self.y[0] + r * pitch * u[1]
        elif self.size() == 3:
            # For 3-pixel events we estimate both r and theta using the probit function.
            eta_sum = _eta[0] + _eta[1]
            eta_diff = (_eta[0] - _eta[1]) / eta_sum
            if eta_sum > p3_r or p3_r <= 0.:
                r = Probit().evaluate(eta_sum, mu3_r, s3_r)
            else:
                y_pivot = Probit().evaluate(p3_r, mu3_r, s3_r)
                r = y_pivot / p3_r * eta_sum
            theta = Probit().evaluate((eta_diff + 1)/2, 0, s3_t) / r
            # Reconstructing the position using r and theta
            x_recon = self.x[0] + r * pitch * (np.cos(theta) * u[0] + np.sin(theta) * v[0])
            y_recon = self.y[0] + r * pitch * (np.cos(theta) * u[1] + np.sin(theta) * v[1])
        else:
            # This condition should never be reached because of the check at the beginning of the
            # method, but it's here for safety.
            raise RuntimeError("Cluster must contain 2 or 3 pixels to reconstruct position with" \
                               " eta function")
        return x_recon, y_recon

    def mle(self, charge_matrix: "MatrixChargeDiffusion",
            sigma_noise: float) -> Tuple[float, float]:
        """Return the cluster reconstructed position using the maximum likelihood estimator. The
        computation is performed using the negative log-likelihood, which is minimized with the
        iminuit package.

        To speed up the computation, the negative log-likelihood and its gradient are implemented
        in the likelihood.py module and decorated with numba.njit.

        Arguments
        ---------
        charge_matrix : MatrixChargeDiffusion
            The charge matrix object containing the charge diffusion map, the gradients and the
            pixel coordinates.
        sigma_noise : float
            The noise level for the likelihood computation.
        """
        f = charge_matrix.eta
        x_bins = charge_matrix.x_bins
        y_bins = charge_matrix.y_bins
        x0, y0 = x_bins[0], y_bins[0]
        dx_bin, dy_bin = x_bins[1]-x_bins[0], y_bins[1]-y_bins[0]
        pha = self.pha

        def nll(x, y):
            """Wrapper around the nll_numba function to be passed to iminuit, which expects a
            function that takes the parameters to be optimized as arguments.
            """
            return nll_numba(x, y, pha, f, x0, y0, dx_bin, dy_bin, sigma_noise)

        def nll_grad(x, y):
            """Wrapper around the nll_grad_numba function to be passed to iminuit, which expects a
            function that takes the parameters to be optimized as arguments.
            """
            return nll_grad_numba(x, y, pha, f, x0, y0, dx_bin, dy_bin, sigma_noise)
        
        x_centroid, y_centroid = self.centroid()
        x_centroid -= self.x[0]
        y_centroid -= self.y[0]
        start_x, start_y = x_centroid / 0.005, y_centroid / 0.005
        m = iminuit.Minuit(nll, x=start_x, y=start_y, grad=nll_grad)
        m.limits = [(x_bins[0], x_bins[-1]), (y_bins[0], y_bins[-1])]
        m.errors = [0.01, 0.01]
        m.migrad()
        m.minos()
        # from aptapy.plotting import plt
        
        # print(f"x_rc = {m.values['x']:.3f} + {m.merrors['x'].upper:.3f} - {m.merrors['x'].lower:.3f}")
        # print(f"y_rc = {m.values['y']:.3f} + {m.merrors['y'].upper:.3f} - {m.merrors['y'].lower:.3f}")
        # x_range = np.linspace(x_bins[0], x_bins[-1], 100)
        # y_range = np.linspace(y_bins[0], y_bins[-1], 100)
        # X, Y = np.meshgrid(x_range, y_range)

        # Z = np.array([[nll(x, y) for x in x_range] for y in y_range])

        # plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        # plt.colorbar(label='NLL')
        # plt.plot(m.values['x'], m.values['y'], 'ro', label='minimum')
        # plt.plot(start_x, start_y, 'wx', label='centroid')             
        # plt.legend()

        return self.x[0] + m.values['x'] * 0.005, self.y[0] + m.values['y'] * 0.005
    
    def position(self):
        """Return the cluster reconstructed position using the position reconstruction algorithm
        specified in the constructor.

        This method is a wrapper around the different position reconstruction algorithms. It checks
        the value of pos_recon_algorithm and calls the corresponding method. If the value of
        pos_recon_algorithm is not recognized, it raises an error.
        """
        if self.pos_recon_algorithm == "centroid":
            return self.centroid()
        if self.pos_recon_algorithm == "eta":
            return self.eta(**self.recon_pars)
        if self.pos_recon_algorithm == "mle":
            return self.mle(**self.recon_pars)
        raise RuntimeError(f"Unknown position reconstruction method {self.pos_recon_algorithm}")

    def pos_error(self, recon_par: str) -> Tuple[float, float]:
        """Return the error of the reconstructed parameter.
        """
        if recon_par == "x":
            return self._errx_low, self._errx_high
        if recon_par == "y":
            return self._erry_low, self._erry_high
        raise ValueError(f"Invalid reconstruction parameter: {recon_par}")


@dataclass
class ClusteringBase:

    """Base class for the clustering.
    """

    readout: HexagonalReadoutBase
    zero_sup_threshold: float

    def __post_init__(self) -> None:
        """Check if the readout gain is a scalar or an array.
        """
        self._scalar_gain = isinstance(self.readout.gain, (int, float))

    def _gain(self, row: np.ndarray, col: np.ndarray) -> np.ndarray:
        """Return the correct gain value for the given row and column indexes.

        This method is necessary to handle both the case of a scalar gain and the case of a gain
        map. It would be a mess to handle the two cases in the run method, so we check the type
        in the constructor and then we return the gain value in a unified way here.
        """
        if self._scalar_gain:
            return self.readout.gain
        return self.readout.gain[row, col]

    def zero_suppress(self, array: np.ndarray) -> np.ndarray:
        """Zero suppress a generic array.
        """
        out = array.copy()
        out[out <= self.zero_sup_threshold] = 0
        return out

    def position_suppress(self, pha: np.ndarray, col: np.ndarray, row: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Suppress pixels in the cluster that do not satisfy the position requirements.

        If the cluster contains 2 or fewer pixels, no action is taken. For clusters with 
        more than 2 pixels, the algorithm retains the two most charged pixels and 
        only one additional neighbor (the one with the highest charge) of the second 
        pixel, discarding all others.

        Arguments
        ---------
        pha : np.ndarray
            The array of pulse heights of the pixels in the cluster, ordered in decreasing order.
        col : np.ndarray
            The array of column indexes of the pixels in the cluster.
        row : np.ndarray
            The array of row indexes of the pixels in the cluster.
        
        Returns
        -------
        pha : np.ndarray
            The array of pulse heights of the pixels in the cluster after position suppression,
            ordered in decreasing order.
        col : np.ndarray
            The array of column indexes of the pixels in the cluster after position suppression,
            ordered in decreasing order of pulse height.
        row : np.ndarray
            The array of row indexes of the pixels in the cluster after position suppression,
            ordered in decreasing order of pulse height.
        """
        # If we have 2 or less pixels above threshold, we can't do anything, just remove the zeros.
        if np.count_nonzero(pha) <= 2 or pha.size <= 2:
            mask = pha > 0
            return pha[mask], col[mask], row[mask]
        # For events with more than 2 pixels above threshold, we keep only the highest neighbor of
        # the second pixel.
        pix = list(zip(col, row))
        # We find the neighbors of the second pixel.
        neighbors = set(self.readout.neighbors(col[1], row[1]))
        # We always keep the first two pixels, plus the two neighbors of the second pixel.
        mask = np.array([True, True] + [p in neighbors for p in pix[2:]], dtype=bool)
        mask = mask & (pha > 0)
        # Throw away the pixels that are not neighbors of the second pixel and that are zero.
        out_pha = pha[mask]
        out_col = col[mask]
        out_row = row[mask]
        # Sort the arrays in decreasing order of pulse height.
        idx = np.argsort(-out_pha)[:3]
        return out_pha[idx], out_col[idx], out_row[idx]


    def run(self, event: DigiEventRectangular) -> Cluster:
        """Workhorse method to be reimplemented by derived classes.
        """
        raise NotImplementedError


@dataclass
class ClusteringNN(ClusteringBase):

    """Neirest neighbor clustering.

    This is a very simple clustering strategy where we use the highest pixel in
    the event as a seed, loop over the six neighbors (after the zero suppression)
    and keep the N highest pixels.

    Arguments
    ---------
    num_neighbors : int
        The number of neighbors (between 0 and 6) to include in the cluster.
    """

    num_neighbors: int
    pos_recon_algorithm: str
    recon_pars: dict = None

    def run(self, event) -> Cluster:
        """Overladed method.

        .. warning::
           The loop ever the neighbors might likely be vectorized and streamlined
           for speed using proper numpy array for the offset indexes.
        """
        if isinstance(event, DigiEventCircular):
            # If the readout is circular, we want to take all the neirest neighbors.
            # Trailing -1 is bc the central px is already considered.
            self.num_neighbors = 6 #HexagonalReadoutCircular.NUM_PIXELS - 1
            col = [event.column]
            row = [event.row]
            adc_channel_order = [self.readout.adc_channel(event.column, event.row)]
            # Taking the NN in logical coordinates ...
            gain_array = [self._gain(event.row, event.column)]
            for _col, _row in self.readout.neighbors(event.column, event.row):
                col.append(_col)
                row.append(_row)
                # ... transforming the coordinates of the NN in its corresponding ADC channel ...
                adc_channel_order.append(self.readout.adc_channel(_col, _row))
                gain_array.append(self._gain(_row, _col))
            # ... reordering the pha array for the correspondance (col[i], row[i]) with pha[i].
            pha = (event.pha[adc_channel_order] - self.readout.offset) / np.array(gain_array)
            # Converting lists into numpy arrays
            col = np.array(col)
            row = np.array(row)
            pha = np.array(pha)
        # pylint: disable = invalid-name
        elif isinstance(event, DigiEventRectangular):
            seed_col, seed_row = event.highest_pixel()
            col = [seed_col]
            row = [seed_row]
            for _col, _row in self.readout.neighbors(seed_col, seed_row):
                col.append(_col)
                row.append(_row)
            col = np.array(col)
            row = np.array(row)
            pha = np.array([(event(_col, _row) - self.readout.offset) / self._gain(_row, _col)
                            for _col, _row in zip(col, row)])
        # Zero suppressing the event (whatever the readout type)...
        pha = self.zero_suppress(pha)
        # Array indexes in order of decreasing pha---note that we use -pha to
        # trick argsort into sorting values in decreasing order.
        idx = np.argsort(-pha)
        # Only pick the seed and the N highest pixels.
        # This is useless for the circular readout because in that case all
        # neighbors are used for track reconstruction.
        mask = idx[:self.num_neighbors + 1]
        # Sort the arrays in decreasing order before applying the position suppression.
        pha, col, row = self.position_suppress(pha[mask], col[mask], row[mask])
        x, y = self.readout.pixel_to_world(col, row)
        return Cluster(x, y, col, row, pha, self.pos_recon_algorithm, self.recon_pars)


@dataclass
class ClusteringHex(ClusteringBase):

    recon_pars: dict = None

    def run(self, event) -> Cluster:
        """Overladed method.

        .. warning::
           The loop ever the neighbors might likely be vectorized and streamlined
           for speed using proper numpy array for the offset indexes.
        """
        # Always take all the six neighbors
        col, row, pha = [], [], []
        if isinstance(event, DigiEventCircular):
            gain_array = [self._gain(event.row, event.column)]
            col.append(event.column)
            row.append(event.row)
            pha.append((event.pha[self.readout.adc_channel(event.column, event.row)] - self.readout.offset) / gain_array[0])
            for _col, _row in self.readout.neighbors(event.column, event.row):
                col.append(_col)
                row.append(_row)
                _pha = event.pha[self.readout.adc_channel(_col, _row)]
                pha.append((_pha - self.readout.offset) / self._gain(_row, _col))
        # pylint: disable = invalid-name
        elif isinstance(event, DigiEventRectangular):
            seed_col, seed_row = event.highest_pixel()
            col.append(seed_col)
            row.append(seed_row)
            pha.append((event(seed_col, seed_row) - self.readout.offset) / self._gain(seed_row, seed_col))
            for _col, _row in self.readout.neighbors(seed_col, seed_row):
                col.append(_col)
                row.append(_row)
                pha.append((event(_col, _row) - self.readout.offset) / self._gain(_row, _col))
        # Converting lists into numpy arrays
        col = np.array(col)
        row = np.array(row)
        pha = np.array(pha)
        # Calculate the physical coordinates of the pixels in the cluster.
        x, y = self.readout.pixel_to_world(col, row)
        return Cluster(x, y, col, row, pha, "mle", self.recon_pars)
