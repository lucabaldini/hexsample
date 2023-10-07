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


def required_args(parser : ArgumentParser) -> list:
    """Return a list of the positional arguments for a given parser.

    This is useful to retrieve all the default values from an ArgumentParser
    object, as calling parse_args([]) will only work if there are not required
    arguments---otherwhise the parser will complain about them missing.

    Arguments
    ---------
    parser : ArgumentParser
        The argument parser object for a given application.
    """
    return [action.dest for action in parser._actions if action.required]

def default_args(parser : ArgumentParser) -> dict:
    """Return the default arguments for a given ArgumentParser object.

    If the parser has no positional arguments, this is simply achieved via a
    ``parse_args([])`` call. In presence of positional arguments things are
    more complicates, as calling ``parse_args([])`` will trigger a parser error.
    In that case we use phony values for the positional arguments just to trick
    the ArgumentParser into not raising an exception, and strip them after the
    fact from the output dictionary.

    .. note::
       The positional arguments (if any), which have no default, are not contained
       in the final dictionary, and must be provided by the user in the
       ``update_args()`` call via keyword arguments.

    Arguments
    ---------
    parser : ArgumentParser
        The argument parser object for a given application.
    """
    args = required_args(parser)
    kwargs = vars(parser.parse_args(args))
    [kwargs.pop(key) for key in args]
    return kwargs

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
