# Copyright (C) 2023 luca.baldini@pi.infn.it
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

"""Reconstruction facilities.
"""

from dataclasses import dataclass
from typing import Tuple

import xraydb

from hexsample.clustering import Cluster


DEFAULT_IONIZATION_POTENTIAL = xraydb.ionization_potential('Si')

@dataclass
class ReconEventBase:

    """Descriptor for a reconstructed event.

    Arguments
    ---------
    trigger_id : int
        The trigger identifier.

    timestamp : float
        The timestamp (in s) of the event.

    livetime : int
        The livetime (in us) since the last event.

    cluster : Cluster
        The reconstructed cluster for the event.
    """

    trigger_id: int
    timestamp: float
    livetime: int
    cluster: Cluster

    def energy(self, ionization_potential: float = DEFAULT_IONIZATION_POTENTIAL) -> float:
        """Return the energy of the event in eV.

        .. warning::
           This is currently using the ionization energy of Silicon to do the
           conversion, assuming a detector gain of 1. We will need to do some
           bookkeeping, here, to make this work reliably.
        """
        return ionization_potential * self.cluster.pulse_height()

    def position(self) -> Tuple[float, float]:
        """Return the reconstructed position of the event.
        """
        return self.cluster.centroid()



@dataclass
class ReconEvent:

    """Descriptor for a reconstructed event.

    Arguments
    ---------
    trigger_id : int
        The trigger identifier.

    timestamp : float
        The timestamp (in s) of the event.

    livetime : int
        The livetime (in us) since the last event.

    roi_size : int
        The ROI size for the event.

    cluster : Cluster
        The reconstructed cluster for the event.
    """

    trigger_id: int
    timestamp: float
    livetime: int
    #roi_size: int
    cluster: Cluster

    def energy(self, ionization_potential: float = DEFAULT_IONIZATION_POTENTIAL) -> float:
        """Return the energy of the event in eV.

        .. warning::
           This is currently using the ionization energy of Silicon to do the
           conversion, assuming a detector gain of 1. We will need to do some
           bookkeeping, here, to make this work reliably.
        """
        return ionization_potential * self.cluster.pulse_height()

    def position(self) -> Tuple[float, float]:
        """Return the reconstructed position of the event.
        """
        return self.cluster.centroid()
