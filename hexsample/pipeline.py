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

"""Pipeline facilities.
"""

import sys

from hexsample import HEXSAMPLE_BIN
from hexsample.app import ArgumentParser

# When we import stuff from the executable scripts (whose folder is not included
# in the $PYTHONPATH) we don't want to pollute the filesystem with assembled bytecode.
sys.path.append(f'{HEXSAMPLE_BIN}')
sys.dont_write_bytecode = 1

# pylint: disable=import-error, wrong-import-position, wrong-import-order
from hxrecon import HXRECON_ARGPARSER, hxrecon as _hxrecon
from hxsim import HXSIM_ARGPARSER, hxsim as _hxsim


def default_args(parser : ArgumentParser) -> dict:
    """Return the default arguments for a given ArgumentParser object.

    Arguments
    ---------
    parser : ArgumentParser
        The argument parser object for a given application.
    """
    return parser.parse_args('').__dict__

def update_args(parser : ArgumentParser, **kwargs) -> dict:
    """Retrieve the default option from an ArgumentParser object and update
    specific keys based on arbitrary keyword arguments.

    Arguments
    ---------
    parser : ArgumentParser
        The argument parser object for a given application.

    kwargs : dict
        Additional keyword arguments.
    """
    args = default_args(parser)
    args.update(kwargs)
    return args

def hxrecon(**kwargs):
    """Application wrapper.
    """
    return _hxrecon(**update_args(HXRECON_ARGPARSER, **kwargs))

def hxsim(**kwargs):
    """Application wrapper.
    """
    return _hxsim(**update_args(HXSIM_ARGPARSER, **kwargs))
