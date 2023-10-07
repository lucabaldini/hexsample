#!/usr/bin/env python
#
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

"""Event display.
"""

from hexsample.app import ArgumentParser
from hexsample.digi import Xpol3
from hexsample.display import HexagonalGridDisplay
from hexsample.io import DigiInputFile
from hexsample.plot import plt


__description__ = \
"""Single event display.
"""

# Parser object.
HXDISPLAY_ARGPARSER = ArgumentParser(description=__description__)
HXDISPLAY_ARGPARSER.add_infile()


def hxdisplay(**kwargs):
    """Application main entry point.
    """
    file_path = kwargs.get('infile')
    display = HexagonalGridDisplay(Xpol3())
    input_file = DigiInputFile(file_path)
    for event in input_file:
        print(event.ascii())
        display.draw_digi_event(event, zero_sup_threshold=0)
        display.show()
    input_file.close()



if __name__ == '__main__':
    hxdisplay(**vars(HXDISPLAY_ARGPARSER.parse_args()))
