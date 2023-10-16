#!/usr/bin/env python
#
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

"""Diff utility for event files.
"""



from hexsample.app import ArgumentParser
from hexsample.fileio import DigiInputFile



__description__ = \
"""Compare two different event files.
"""

# Parser object.
HXDIFF_ARGPARSER = ArgumentParser(description=__description__)
HXDIFF_ARGPARSER.add_argument('infiles', type=str, nargs=2,
            help='path to the two input files to be compared')



def hxdiff(**kwargs):
    """Application main entry point.
    """
    file_path1, file_path2 = kwargs['infiles']
    file1 = DigiInputFile(file_path1)
    file2 = DigiInputFile(file_path2)
    #for i in range(10):
        #evt1 = file1.digi_event(i)
        #evt2 = file2.digi_event(i)
    for evt1, evt2 in zip(file1, file2):
        print(evt1)
        print(evt2)
        #print(evt1 == evt2)
        input()


if __name__ == '__main__':
    hxdiff(**vars(HXDIFF_ARGPARSER.parse_args()))
