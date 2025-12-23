# Copyright (C) 2025 the hexsample team.
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

"""Test suite for hexsample.base
"""

from dataclasses import dataclass

import pytest

from hexsample.base import type_proxy

# A few helper classes to help test things without having to import other modules
# that might evolve in the future.

@dataclass
class _Line:
    energy: float = 6000.


@dataclass
class _LineForest:
    element: str = "Cu"
    initial_level: str = "K"


class _Vanilla:
    pass



def test_type_proxy_decorator():
    """Test for the type_proxy class decorator.
    """

    @type_proxy()
    class TypeA:

        _KEY = "spectrum"
        _PROXY_DICT = {
            "line": _Line,
            "forest": _LineForest,
        }

    assert TypeA.choices() == ("line", "forest")
    assert TypeA.default() == "line"

    obj = TypeA.factory()
    assert isinstance(obj, _Line)
    assert obj.energy == _Line.energy

    obj = TypeA.factory(spectrum="line", energy=2000.)
    assert isinstance(obj, _Line)
    assert obj.energy == 2000.

    obj = TypeA.factory(spectrum="forest")
    assert isinstance(obj, _LineForest)

    @type_proxy(default="forest")
    class TypeB:

        _KEY = "spectrum"
        _PROXY_DICT = {
            "line": _Line,
            "forest": _LineForest,
        }

    assert TypeB.choices() == ("line", "forest")
    assert TypeB.default() == "forest"

    with pytest.raises(ValueError):

        @type_proxy
        class TypeC:

            _KEY = "spectrum"
            _PROXY_DICT = {
                "vanilla": _Vanilla,  # Not a dataclass
            }

    with pytest.raises(ValueError):

        @type_proxy
        class TypeD:

            # Missing _KEY and _PROXY_DICT
            pass