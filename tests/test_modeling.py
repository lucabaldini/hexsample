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

"""Test suite for modeling.py
"""

from hexsample.modeling import Constant, Line, Gaussian, PowerLaw, Exponential
from hexsample.plot import plt, setup_gca


def _test_model(model):
    """Basic test function.
    """
    plt.figure(model.name())
    model.plot()
    model.stat_box()
    setup_gca(xlabel='x [a. u.]', ylabel='f(x)', grids=True)

def test_models():
    """Test all models.
    """
    for cls in Constant, Line, Gaussian, PowerLaw, Exponential:
        model = cls()
        _test_model(model)



if __name__ == '__main__':
    test_models()
    plt.show()
