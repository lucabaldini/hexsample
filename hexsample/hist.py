# Copyright (C) 2022 luca.baldini@pi.infn.it
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""Histogram facilities.
"""

from __future__ import annotations

import numbers
from typing import Optional, Tuple

import matplotlib
import numpy as np

from hexsample.plot import plt, setup_gca


# pylint: disable=invalid-name, too-many-arguments, attribute-defined-outside-init, too-many-instance-attributes


class HistogramBase:

    """Base class for an n-dimensional histogram.

    This interface to histograms is profoundly different for the minimal
    numpy/matplotlib approach, where histogramming methods return bare
    vectors of bin edges and counts. The main underlying ideas are

    * we keep track of the bin contents, the bin entries and the sum of the
      weights squared (the latter for the purpose of calculating the errors);
    * we control the axis label and the plotting styles;
    * we provide two separate interfaces, fill() and set_content(), to
      fill the histogram from either unbinned or binned data;
    * we support the basic arithmetics (addition, subtraction and multiplication
      by a scalar);
    * we support full data persistence (I/O) in FITS format.

    Note that this base class is not meant to be instantiated directly, and
    the interfaces to concrete histograms of specific dimensionality are
    defined in the sub-classes.

    Parameters
    ----------
    binning : n-tuple of array
        the bin edges on the different axes.

    labels : n-tuple of strings
        the text labels for the different axes.
    """

    PLOT_OPTIONS = {}

    def __init__(self, binning : Tuple[np.array], labels : Tuple[str]) -> None:
        """Constructor.
        """
        assert len(labels) == len(binning) + 1
        self.binning = tuple(binning)
        self.labels = list(labels)
        self.shape = tuple(len(bins) - 1 for bins in self.binning)
        self.num_axes = len(self.binning)
        self.content = np.zeros(shape=self.shape, dtype=float)
        self.entries = np.zeros(shape=self.shape, dtype=float)
        self.sumw2 = np.zeros(shape=self.shape, dtype=float)

    def fill(self, *data : np.array, weights : Optional[np.array] = None) -> HistogramBase:
        """Fill the histogram from unbinned data.

        Note this method is returning the histogram instance, so that the function
        call can be chained.
        """
        if weights is None:
            weights = np.ones(data[0].shape, dtype=float)
        elif isinstance(weights, numbers.Number):
            weights = np.full(data[0].shape, weights, dtype=float)
        data = np.vstack(data).T
        content, _ = np.histogramdd(data, bins=self.binning, weights=weights)
        entries, _ = np.histogramdd(data, bins=self.binning)
        sumw2, _ = np.histogramdd(data, bins=self.binning, weights=weights**2.)
        self.content += content
        self.entries += entries
        self.sumw2 += sumw2
        return self

    def set_content(self, content : np.array, entries : Optional[np.array] = None,
                    errors : Optional[np.array] = None) -> HistogramBase:
        """Set the bin contents programmatically from binned data.

        Note this method is returning the histogram instance, so that the function
        call can be chained.
        """
        assert content.shape == self.shape
        self.content = content
        if entries is not None:
            assert entries.shape == self.shape
            self.entries = entries
        if errors is not None:
            self.set_errors(errors)
        return self

    def errors(self) -> np.array:
        """Return the bin errors.
        """
        return np.sqrt(self.sumw2)

    def set_errors(self, errors : np.array) -> None:
        """Set the bin errors.
        """
        assert errors.shape == self.shape
        self.sumw2 = errors**2.

    def bin_centers(self, axis : int = 0) -> np.array:
        """Return the bin centers for a specific axis.
        """
        return 0.5 * (self.binning[axis][1:] + self.binning[axis][:-1])

    def bin_widths(self, axis : int = 0) -> np.array:
        """Return the bin widths for a specific axis.
        """
        return np.diff(self.binning[axis])

    @staticmethod
    def bisect(binning, values : np.array, side : str = 'left') -> int:
        """Return the indices corresponding to a given array of values for a
        given binning.
        """
        return np.searchsorted(binning, values, side) - 1

    def find_bin(self, *coords : float) -> Tuple[int]:
        """Find the bin corresponding to a given set of "physical" coordinates
        on the histogram axes.

        This returns a tuple of integer indices that can be used to address
        the histogram content.
        """
        return tuple(self.bisect(binning, value) for binning, value in zip(self.binning, coords))

    def find_bin_value(self, *coords : float) -> float:
        """Find the histogram content corresponding to a given set of "physical"
        coordinates on the histogram axes.
        """
        return self.content[self.find_bin(*coords)]

    def empty_copy(self) -> HistogramBase:
        """Create an empty copy of a histogram.
        """
        return self.__class__(*self.binning, *self.labels)

    def copy(self) -> HistogramBase:
        """Create a full copy of a histogram.
        """
        hist = self.empty_copy()
        hist.set_content(self.content.copy(), self.entries.copy())
        return hist

    def __add__(self, other : HistogramBase) -> HistogramBase:
        """Histogram addition.
        """
        hist = self.empty_copy()
        hist.set_content(self.content + other.content, self.entries + other.entries,
            self.sumw2 + other.sumw2)
        return hist

    def __sub__(self, other : HistogramBase) -> HistogramBase:
        """Histogram subtraction.
        """
        hist = self.empty_copy()
        hist.set_content(self.content - other.content, self.entries + other.entries,
            self.sumw2 + other.sumw2)
        return hist

    def __mul__(self, value : np.array) -> HistogramBase:
        """Histogram multiplication by a scalar.
        """
        hist = self.empty_copy()
        hist.set_content(self.content * value, self.entries, self.errors() * value)
        return hist

    def __rmul__(self, value : np.array) -> HistogramBase:
        """Histogram multiplication by a scalar.
        """
        return self.__mul__(value)

    def set_axis_label(self, axis : int, label : str) -> None:
        """Set the label for a given axis.
        """
        self.labels[axis] = label

    def _plot(self, **kwargs) -> None:
        """No-op plot() method, to be overloaded by derived classes.
        """
        raise NotImplementedError(f'_plot() not implemented for {self.__class__.__name__}' )

    def plot(self, **kwargs) -> None:
        """Plot the histogram.
        """
        for key, value in self.PLOT_OPTIONS.items():
            kwargs.setdefault(key, value)
        self._plot(**kwargs)
        setup_gca(xmin=self.binning[0][0], xmax=self.binning[0][-1],
                  xlabel=self.labels[0], ylabel=self.labels[1])



class Histogram1d(HistogramBase):

    """Container class for one-dimensional histograms.
    """

    PLOT_OPTIONS = dict(lw=1.25, alpha=0.4, histtype='stepfilled')

    def __init__(self, xbins, xlabel='', ylabel='Entries/bin'):
        """Constructor.
        """
        HistogramBase.__init__(self, (xbins, ), [xlabel, ylabel])

    def _plot(self, **kwargs):
        """Overloaded make_plot() method.
        """
        plt.hist(self.bin_centers(0), self.binning[0], weights=self.content, **kwargs)



class Histogram2d(HistogramBase):

    """Container class for two-dimensional histograms.
    """

    PLOT_OPTIONS = dict(cmap=plt.get_cmap('hot'))

    def __init__(self, xbins : np.array, ybins : np.array, xlabel : str = '',
                 ylabel : str = '', zlabel : str = 'Entries/bin') -> None:
        """Constructor.
        """
        HistogramBase.__init__(self, (xbins, ybins), [xlabel, ylabel, zlabel])

    def _plot(self, logz : bool = False, **kwargs) -> None:
        """Overloaded make_plot() method.
        """
        x, y = (v.flatten() for v in np.meshgrid(self.bin_centers(0), self.bin_centers(1)))
        bins = self.binning
        w = self.content.T.flatten()
        if logz:
            # Hack for a deprecated functionality in matplotlib 3.3.0
            # Parameters norm and vmin/vmax should not be used simultaneously
            # If logz is requested, we intercent the bounds when created the norm
            # and refrain from passing vmin/vmax downstream.
            vmin = kwargs.pop('vmin', None)
            vmax = kwargs.pop('vmax', None)
            kwargs.setdefault('norm', matplotlib.colors.LogNorm(vmin, vmax))
        plt.hist2d(x, y, bins, weights=w, **kwargs)
        colorbar = plt.colorbar()
        if self.labels[2] is not None:
            colorbar.set_label(self.labels[2])



class Histogram3d(HistogramBase):

    """Container class for three-dimensional histograms.
    """

    # pylint: disable = abstract-method

    def __init__(self, xbins : np.array, ybins : np.array, zbins : np.array,
                 xlabel : str = '', ylabel : str = '', zlabel : str = '',
                 wlabel : str = 'Entries/bin') -> None:
        """Constructor.
        """
        HistogramBase.__init__(self, (xbins, ybins, zbins), [xlabel, ylabel, zlabel, wlabel])
