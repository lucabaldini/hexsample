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

"""Monte Carlo facilities.
"""


from dataclasses import dataclass

import numpy as np

from hexsample.rng import rng
from hexsample.sensor import Sensor
from hexsample.source import Source


@dataclass
class MonteCarloEvent:

    """Descriptor for the ground truth of a simulated event.

    Arguments
    ---------
    timestamp : float
        The timestamp for the event in s.

    energy : float
        The energy of the event in eV.

    x : float
        The x coordinate of the photon absorption point in cm.

    y : float
        The y coordinate of the photon absorption point in cm.

    z : float
        The z coordinate of the photon absorption point in cm.

    num_pairs : int
        The number of electron-hole pairs created in the detector active volume.
    """

    timestamp : float
    energy : float
    absx : float
    absy : float
    absz : float
    num_pairs : int

    def propagate(self, diffusion_sigma : float):
        """Propagate the primary ionization down to the readout plane.
        """
        # pylint: disable=invalid-name
        sigma = diffusion_sigma / 10000. * np.sqrt(self.absz)
        x = rng.normal(self.absx, sigma, size=self.num_pairs)
        y = rng.normal(self.absy, sigma, size=self.num_pairs)
        return x, y



class PhotonList:

    """Small convenience class representing a simulated photon list.

    Arguments
    ---------
    source : Source
        The X-ray source.

    sensor : Sensor
        The X-ray sensor.

    num_photons : int
        The number of photons to be simulated.
    """

    def __init__(self, source : Source, sensor : Sensor, num_photons : int) -> None:
        """Constructor.
        """
        self.timestamp, self.energy, self.absx, self.absy = source.rvs(num_photons)
        self.absz = sensor.rvs_absz(self.energy)
        self.num_pairs = sensor.material.rvs_num_pairs(self.energy)
        self.__index = -1

    def __iter__(self):
        """Overloaded method for the implementation of the iterator protocol.
        """
        self.__index = -1
        return self

    def __next__(self) -> MonteCarloEvent:
        """Overloaded method for the implementation of the iterator protocol.
        """
        self.__index += 1
        if self.__index == len(self.timestamp):
            raise StopIteration
        args = [item[self.__index] for item in \
            (self.timestamp, self.energy, self.absx, self.absy, self.absz, self.num_pairs)]
        return MonteCarloEvent(*args)
