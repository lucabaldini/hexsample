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

from hexsample.base import TypeProxy

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
    """Test for the TypeProxy class.
    """
    proxy = TypeProxy("spectrum")
    proxy.register("line", _Line)
    proxy.register("forest", _LineForest)
    # We actively prevent registering non-dataclass types.
    with pytest.raises(TypeError):
        proxy.register("vanilla", _Vanilla)
    # And we do not allow re-registering existing keys.
    with pytest.raises(ValueError):
        proxy.register("line", _Line)
    # Basic functionality.
    assert proxy.key() == "spectrum"
    assert proxy.choices() == ("line", "forest")
    assert proxy.default() == "line"
    # Create objects.
    spectrum = proxy.create("line")
    assert isinstance(spectrum, _Line)
    assert spectrum.energy == _Line.energy
    spectrum = proxy.create("line", energy=2000.)
    assert isinstance(spectrum, _Line)
    assert spectrum.energy == 2000.
    spectrum = proxy.create("forest")
    assert isinstance(spectrum, _LineForest)
    assert spectrum.element == _LineForest.element
    assert spectrum.initial_level == _LineForest.initial_level
