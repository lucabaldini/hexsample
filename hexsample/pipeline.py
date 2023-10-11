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


def required_arguments(parser : ArgumentParser) -> list:
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

def default_arguments(parser : ArgumentParser) -> dict:
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

    Returns
    -------
    args, kwargs : list, dict
        The list of positional arguments and a dictionary of all the other arguments.
    """
    # Positional arguments.
    args = required_arguments(parser)
    # All parser arguments.
    kwargs = vars(parser.parse_args(args))
    # Strip the positional arguments from the complete list.
    [kwargs.pop(key) for key in args]
    # And return the two sets separately.
    return args, kwargs

def update_arguments(parser : ArgumentParser, **kwargs) -> dict:
    """Retrieve the default option from an ArgumentParser object and update
    specific keys based on arbitrary keyword arguments.

    Arguments
    ---------
    parser : ArgumentParser
        The argument parser object for a given application.

    kwargs : dict
        Additional keyword arguments.
    """
    # Retrieve the default arguments.
    _args, _kwargs = default_arguments(parser)
    # Loop over the kwargs passed to the function to make sure that all of them
    # are recognized by the parser.
    for key in kwargs:
        if key not in _args and key not in _kwargs:
            raise RuntimeError(f'Unknown parameter {key} passed to a pipeline component')
    _kwargs.update(kwargs)
    return _kwargs

def hxrecon(**kwargs):
    """Application wrapper.
    """
    return _hxrecon(**update_arguments(HXRECON_ARGPARSER, **kwargs))

def hxsim(**kwargs):
    """Application wrapper.
    """
    return _hxsim(**update_arguments(HXSIM_ARGPARSER, **kwargs))
