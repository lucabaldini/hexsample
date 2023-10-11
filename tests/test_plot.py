# Copyright (C) 2022 luca.baldini@pi.infn.it
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

"""Test suite for plot.py
"""

from hexsample.plot import plt, PlotCard


def test_card():
    """Test for the plot cards.
    """
    card = PlotCard()
    card.add_string('Label', 'Content')
    card.add_blank()
    card.add_quantity('Fixed float', 1.0)
    card.add_quantity('Formatted fixed float', 1.0, fmt='.5f')
    card.add_quantity('Fixed int', 1)
    card.add_quantity('Parameter 1', 1.23456, 0.53627)
    card.add_quantity('Fixed float', 1.0, units='cm')
    card.add_quantity('Fixed int', 1, units='cm')
    card.add_quantity('Parameter 1', 1.23456, 0.53627, units='cm')
    card.plot()
    card.plot(0.05, 0.95, ha='left')


if __name__ == '__main__':
    test_card()
    plt.show()
