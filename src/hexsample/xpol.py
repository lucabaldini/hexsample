# Copyright (C) 2022--2023 luca.baldini@pi.infn.it
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

"""Quantities related to the XPOL readout chip.
"""

from hexsample.hexagon import HexagonalLayout
from hexsample.roi import Padding


# Chip size for the two generations.
XPOL1_SIZE = (300, 352)
XPOL1_LAYOUT = HexagonalLayout.EVEN_R
XPOL3_SIZE = (304, 352)
XPOL3_LAYOUt = HexagonalLayout.ODD_R

# Pixel pitch in cm.
XPOL_PITCH = 0.005

# Convenience constants for the XPOL1 default paddings.
XPOL1_SMALL_PADDING = Padding(10, 8)
XPOL1_LARGE_PADDING = Padding(20, 16)
