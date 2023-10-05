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

    def __init__(self, prog : str = None, usage : str = None, description : str = None) -> None:
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

    def add_source_options(self) -> None:
        """Add an option group for the source properties.
        """
        group = self.add_argument_group('source', 'X-ray source properties')
        group.add_argument('--srcelement', type=str, default='Cu',
            help='element generating the line forest')
        group.add_argument('--srclevel', type=str, default='K',
            help='initial level for the line forest')
        group.add_argument('--srcposx', type=float, default=0.,
            help='x position of the source centroid in cm')
        group.add_argument('--srcposy', type=float, default=0.,
            help='y position of the source centroid in cm')
        group.add_argument('--srcsigma', type=float, default=0.1,
            help='one-dimensional standard deviation of the gaussian beam in cm')

    def add_sensor_options(self) -> None:
        """Add an option group for the sensor properties.
        """
        group = self.add_argument_group('sensor', 'Sensor properties')
        group.add_argument('--actmedium', type=str, choices=('Si',), default='Si',
            help='active sensor material')
        group.add_argument('--thickness', type=float, default=0.03,
            help='thickness in cm')
        group.add_argument('--fano', type=float, default=0.116,
            help='fano factor')
        group.add_argument('--transdiffsigma', type=float, default=40.,
            help='diffusion sigma in um per sqrt(cm)')

    def add_readout_options(self) -> None:
        """Add an option group for the readout properties.
        """
        group = self.add_argument_group('readout', 'Redout configuration')
        group.add_argument('--noise', type=float, default=20.,
            help='equivalent noise charge rms in electrons')
        group.add_argument('--gain', type=float, default=1.,
            help='conversion factors between electron equivalent and ADC counts')
        group.add_argument('--offset', type=int, default=0,
            help='optional signal offset in ADC counts')
        group.add_argument('--trgthreshold', type=float, default=250.,
            help='trigger threshold in electron equivalent')
        group.add_argument('--zsupthreshold', type=int, default=0,
            help='zero-suppression threshold in ADC counts')
        group.add_argument('--padding', type=int, nargs=4, default=(2, 2, 2, 2),
            help='padding on the four sides of the ROT')
