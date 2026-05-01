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

"""Quantities related to the family of XPOL readout chips.
"""

from dataclasses import dataclass
from typing import Tuple

from .hexagon import HexagonalLayout
from .roi import Padding


@dataclass(frozen=True)
class XPOL1:

    """XPOL1 chip properties.
    """

    size: Tuple[int, int] = (300, 352)
    layout: HexagonalLayout = HexagonalLayout.EVEN_R
    pitch: float = 0.005
    small_padding: Padding = Padding(10, 8, 10, 8)
    large_padding: Padding = Padding(20, 16, 20, 16)


@dataclass(frozen=True)
class XPOL3:

    """XPOL3 chip properties.
    """

    size: Tuple[int, int] = (304, 352)
    layout: HexagonalLayout = HexagonalLayout.ODD_R
    pitch: float = 0.005



_XPOL_DICT = {
    "xpol1": XPOL1,
    "xpol3": XPOL3,
}


def chip_names() -> Tuple[str, ...]:
    """Return a tuple containing all the possible XPOL chip names.
    """
    return tuple(_XPOL_DICT.keys())


def chip_descriptor(name: str):
    """Return the XPOL chip readout descriptor corresponding to the given name.
    """
    try:
        return _XPOL_DICT[name]
    except KeyError as err:
        raise ValueError(f"Unknown XPOL chip name: {name!r}. "
            f"Valid names are: {', '.join(chip_names())}.") from err
