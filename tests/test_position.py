# Copyright (C) 2023--2026 the hexsample team.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Test suite for hexsample.position
"""

import numpy as np

from hexsample.position import eta_2pix, eta_3pix


def test_eta_2pix():
    """Test the eta_2pix function.
    """
    pha = np.array([50., 50.])
    x = np.array([0., 1.])
    y = np.array([0., 0.])
    two_pix_rad_sigma = 0.1
    dx, dy = eta_2pix(pha, x, y, two_pix_rad_sigma)
    assert np.isclose(dx, 0.5, atol=0.01)
    assert np.isclose(dy, 0., atol=0.01)
    x = np.array([0., 0.])
    y = np.array([0., 1.])
    dx, dy = eta_2pix(pha, x, y, two_pix_rad_sigma)
    assert np.isclose(dx, 0., atol=0.01)
    assert np.isclose(dy, 0.5, atol=0.01)


def test_eta_3pix():
    """Test the eta_3pix function.
    """
    pha = np.array([25., 25., 25.])
    x = np.array([0., 1., 0.])
    y = np.array([0., 0., 1.])
    three_pix_rad_offset = np.sqrt(2) / 2
    three_pix_rad_sigma = 0.1
    three_pix_theta_sigma = 0.1
    args = three_pix_rad_offset, three_pix_rad_sigma, three_pix_theta_sigma
    dx, dy = eta_3pix(pha, x, y, *args)
    assert np.isclose(dx, 0.5, atol=0.01)
    assert np.isclose(dy, 0.5, atol=0.01)
