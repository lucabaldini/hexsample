# Copyright (C) 2023 luca.baldini@pi.infn.it
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

"""Test suite for hexsample.roi
"""


from hexsample.roi import Padding, RegionOfInterest, CircularRegionOfInterest


def test_padding(top : int = 2, right : int = 4, bottom : int = 3, left : int = 5) -> None:
    """Test the padding class.
    """
    # With one argument, the padding on the four sides is the same.
    pad = Padding(top)
    print(pad)
    assert pad.top == top
    assert pad.right == top
    assert pad.bottom == top
    assert pad.left == top
    assert tuple(pad) == (top, top, top, top)
    # With two arguments, bottom = top and left = right.
    pad = Padding(top, right)
    print(pad)
    assert pad.top == top
    assert pad.right == right
    assert pad.bottom == top
    assert pad.left == right
    assert tuple(pad) == (top, right, top, right)
    # With three arguments, left = right.
    pad = Padding(top, right, bottom)
    print(pad)
    assert pad.top == top
    assert pad.right == right
    assert pad.bottom == bottom
    assert pad.left == right
    assert tuple(pad) == (top, right, bottom, right)
    # And, finally: different padding on all four sides.
    pad = Padding(top, right, bottom, left)
    print(pad)
    assert pad.top == top
    assert pad.right == right
    assert pad.bottom == bottom
    assert pad.left == left
    assert tuple(pad) == (top, right, bottom, left)

def test_padding_equality():
    """Test the equality opetator for padding.
    """
    pad1 = Padding(2)
    pad2 = Padding(2, 2, 2, 2)
    pad3 = Padding(2, 1, 2, 1)
    assert pad1 == pad2
    assert pad1 != pad3

def test_roi(min_col : int = 0, max_col : int = 5, min_row : int = 25,
    max_row : int = 30, padding : Padding = Padding(2)):
    """Unit test for the RegionOfInterest class.
    """
    roi = RegionOfInterest(min_col, max_col, min_row, max_row, padding)
    print(roi)
    assert roi.min_col == min_col
    assert roi.max_col == max_col
    assert roi.min_row == min_row
    assert roi.max_row == max_row
    assert roi.padding == padding
    num_cols = max_col - min_col + 1
    num_rows = max_row - min_row + 1
    assert roi.num_cols == num_cols
    assert roi.num_rows == num_rows
    assert roi.size == num_cols * num_rows
    assert roi.shape() == (num_rows, num_cols)
    print(roi.col_indexes())
    print(roi.row_indexes())
    print(roi.serial_readout_coordinates())
    print(roi.serial_readout_indexes())
    print(roi.rot_slice())
    print(roi.rot_mask())

def test_circular_roi():
    """Unit test for CircularRegionOfInterest class.
    """
    circ_roi = CircularRegionOfInterest(1, 1)
    print(circ_roi)
    print(circ_roi.at_border((5,5)))
    circ_roi_at_border = CircularRegionOfInterest(5, 5)
    print(circ_roi_at_border)
    print(circ_roi_at_border.at_border((5,5)))


def test_roi_comparison():
    """Test the equality operator for ROI objects.
    """
    roi1 = RegionOfInterest(10, 23, 20, 33, Padding(2))
    roi2 = RegionOfInterest(10, 23, 20, 33, Padding(2, 2, 2, 2))
    roi3 = RegionOfInterest(10, 13, 20, 23, Padding(0, 0, 0, 0))
    assert roi1 == roi2
    assert roi1 != roi3
