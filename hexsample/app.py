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

"""Application utilities.
"""

import argparse

from hexsample import PACKAGE_NAME


START_MESSAGE = f'Welcome to {PACKAGE_NAME}'


def print_start_msg():
    """Print the start message.
    """
    print(START_MESSAGE)



class Formatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):

    """Do nothing class combining our favorite formatting for the
    command-line options, i.e., the newlines in the descriptions are
    preserved and, at the same time, the argument defaults are printed
    out when the --help options is passed.

    The inspiration for this is coming from one of the comments in
    https://stackoverflow.com/questions/3853722
    """


class ArgumentParser(argparse.ArgumentParser):

    """Light-weight wrapper over the argparse ArgumentParser class.

    This is mainly intended to reduce boilerplate code and guarantee a minimum
    uniformity in terms of how the command-line options are expressed across
    different applications.
    """

    def __init__(self, prog : str = None, usage : str = None, description : str = None):
        """Constructor.
        """
        super().__init__(prog, usage, description, formatter_class=Formatter)

    def parse_args(self, *args):
        """Overloaded method.
        """
        print_start_msg()
        return super().parse_args(*args)

    def add_infile(self) -> None:
        """Add an option for the input file.
        """
        help = 'path to the input file'
        self.add_argument('infile', type=str, help=help)

    def add_numevents(self, default : int) -> None:
        """Add an option for the number of events.
        """
        help = 'number of events'
        self.add_argument('--numevents', '-n', type=int, default=default, help=help)

    def add_outfile(self, default : str) -> None:
        """Add an option for the output file.
        """
        help = 'path to the output file'
        self.add_argument('--outfile', '-o', type=str, default=default, help=help)

    def add_trgthreshold(self, default : float = 250.) -> None:
        """Add an option for the trigger threshold.
        """
        help = 'trigger threshold [electron equivalent]'
        self.add_argument('--trgthreshold', '-t', type=float, default=default, help=help)

    def add_zsupthreshold(self, default : int = 0) -> None:
        """Add an option for the zero-suppression threshold.
        """
        help = 'zero-suppression threshold [ADC counts]'
        self.add_argument('--zsupthreshold', '-z', type=float, default=default, help=help)
