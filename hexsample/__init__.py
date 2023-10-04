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

"""System-wide facilities.
"""

from pathlib import Path
import sys

from loguru import logger


# Logger setup.
DEFAULT_LOGURU_HANDLER = dict(sink=sys.stderr, colorize=True,
    format=">>> <level>{message}</level>")

PACKAGE_NAME = 'hexsample'

# Basic package structure.
HEXSAMPLE_ROOT = Path(__file__).parent
HEXSAMPLE_BASE = HEXSAMPLE_ROOT.parent
HEXSAMPLE_DOCS = HEXSAMPLE_BASE / 'docs'
HEXSAMPLE_DOCS_FIGURES = HEXSAMPLE_DOCS / 'figures'
HEXSAMPLE_DOCS_STATIC = HEXSAMPLE_DOCS / '_static'
HEXSAMPLE_TEST = HEXSAMPLE_BASE / 'tests'
HEXSAMPLE_TEST_DATA = HEXSAMPLE_TEST / 'data'

# Make room for the output data.
HEXSAMPLE_DATA = Path.home() / 'hexsampledata'
if not Path.exists(HEXSAMPLE_DATA):
    logger.info(f'Creating data folder {HEXSAMPLE_DATA}...')
    Path.mkdir(HEXSAMPLE_DATA)
