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

import pathlib


from hexsample import logger
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
    file_name1 = pathlib.Path(file_path1).name
    file_name2 = pathlib.Path(file_path2).name
    file1 = DigiInputFile(file_path1)
    file2 = DigiInputFile(file_path2)
    num_differences = 0
    for i, (evt1, evt2) in enumerate(zip(file1, file2)):
        if evt2 != evt1:
            logger.error(f'Mismatch at line {i} of the input files')
            logger.info(f'Event from {file_name1}: {evt1}')
            logger.info(f'Event from {file_name2}: {evt2}')
            num_differences += 1
    if num_differences > 0:
        logger.error(f'Differences found for {num_differences} rows.')
    else:
        logger.info(f'No differences found, all good :-)')



if __name__ == '__main__':
    hxdiff(**vars(HXDIFF_ARGPARSER.parse_args()))
