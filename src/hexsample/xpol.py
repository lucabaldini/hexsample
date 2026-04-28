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

"""Quantities related to the XPOL readout chip.
"""

from enum import Enum
from typing import Tuple

from .hexagon import HexagonalLayout
from .roi import Padding


class XPOL(str, Enum):

    """Enum class expressing the possible XPOL readout chip models.
    """

    XPOL1 = "xpol1"
    XPOL3 = "xpol3"

    @classmethod
    def values(cls) -> Tuple[str, ...]:
        """Return a tuple with all the enum values.
        """
        return tuple(item.value for item in cls)    


# Chip size for the two generations.
XPOL1_SIZE = (300, 352)
XPOL1_LAYOUT = HexagonalLayout.EVEN_R

XPOL3_SIZE = (304, 352)
XPOL3_LAYOUT = HexagonalLayout.ODD_R

# Pixel pitch in cm.
XPOL_PITCH = 0.005

# Convenience constants for the XPOL1 default paddings.
XPOL1_SMALL_PADDING = Padding(10, 8, 10, 8)
XPOL1_LARGE_PADDING = Padding(20, 16, 20, 16)


XPOL_CHIP_DICT = {
    XPOL.XPOL1: XPOL1_SIZE,
    XPOL.XPOL3: XPOL3_SIZE,
}