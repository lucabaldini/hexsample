# Copyright (C) 2025 the hexsample team.
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

"""Main command-line interface.
"""

import argparse
from tokenize import group

from hexsample import __name__ as __package_name__, __version__
from hexsample import hexagon, sensor, source, tasks


def start_message() -> None:
    """Print the start message.
    """
    msg = f"""
    This is {__package_name__} version {__version__}.

    Copyright (C) 2023--2025, the {__package_name__} team.

    {__package_name__} comes with ABSOLUTELY NO WARRANTY.
    This is free software, and you are welcome to redistribute it under certain
    conditions. See the LICENSE file for details.

    Visit https://github.com/lucabaldini/{__package_name__} for more information.
    """
    print(msg)


class _Formatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):

    """Do nothing class combining our favorite formatting for the
    command-line options, i.e., the newlines in the descriptions are
    preserved and, at the same time, the argument defaults are printed
    out when the --help options is passed.

    The inspiration for this is coming from one of the comments in
    https://stackoverflow.com/questions/3853722
    """


class CliArgumentParser(argparse.ArgumentParser):

    """Application-wide argument parser.
    """

    _DESCRIPTION = None
    _EPILOG = None
    _FORMATTER_CLASS = _Formatter

    def __init__(self) -> None:
        """Overloaded method.
        """
        super().__init__(description=self._DESCRIPTION, epilog=self._EPILOG,
                         formatter_class=self._FORMATTER_CLASS)
        subparsers = self.add_subparsers(required=True, help="sub-command help")
        # See https://stackoverflow.com/questions/8757338/
        subparsers._parser_class = argparse.ArgumentParser

        # Run a simulation?
        simulate = subparsers.add_parser("simulate",
            help="run a simulation",
            formatter_class=self._FORMATTER_CLASS)
        self.add_source_options(simulate)
        self.add_sensor_options(simulate)
        self.add_readout_options(simulate)
        simulate.set_defaults(runner=tasks.simulate)

        # Run the event reconstruction?
        recon = subparsers.add_parser("recon",
            help="run the event reconstruction",
            formatter_class=self._FORMATTER_CLASS)

        # Run the single-event display?
        display = subparsers.add_parser("display",
            help="run the single-event display",
            formatter_class=self._FORMATTER_CLASS)

    @staticmethod
    def add_source_options(parser: argparse.ArgumentParser) -> None:
        """Add an option group for to a given (sub-)parser to define the basic
        properties of the X-ray source to be used in a simulation.

        This is clearly suboptimal, as different source spectra and/or beams
        require different sets of parameters, while we are listing all of the
        in no particular order. A more sophisticated approach would require
        multiple parsing stages, which is out of scope for the time being.
        """
        group = parser.add_argument_group("source", "X-ray source properties")
        # Spectral part...
        group.add_argument("--spectrum", type=str, choices=source.spectrum_types(),
                           default=source.default_spectrum_type(),
                           help="spectrum of the X-ray source")
        group.add_argument("--energy", type=float, default=source.LineSpec.energy,
                           help="line energy in eV")
        group.add_argument("--element", type=str, default=source.LineForestSpec.element,
                           help="element generating the line forest")
        group.add_argument("--initial_level", type=str, default=source.LineForestSpec.initial_level,
                           help="initial level for the line forest")
        # ... morphological part...
        group.add_argument("--beam", type=str, choices=source.beam_types(),
                           default=source.default_beam_type(),
                           help="beam shape of the X-ray source")
        group.add_argument("--x0", type=float, default=source.PointBeamSpec.x0,
                           help="x-coordinate of the beam centroid in cm")
        group.add_argument("--y0", type=float, default=source.PointBeamSpec.y0,
                           help="y-coordinate of the beam centroid in cm")
        group.add_argument("--radius", type=float, default=source.DiskBeamSpec.radius,
                           help="radius of the disk beam in cm")
        group.add_argument("--sigma", type=float, default=source.GaussianBeamSpec.sigma,
                           help="standard deviation of the gaussian beam in cm")
        # ... and overall rate.
        group.add_argument("--rate", type=float, default=source.Source.rate,
                           help="overall source rate in photons/s")

    @staticmethod
    def add_sensor_options(parser: argparse.ArgumentParser) -> None:
        """Add an option group for the sensor properties.
        """
        group = parser.add_argument_group("sensor", "Sensor properties")
        group.add_argument("--material_symbol", type=str, choices=sensor.material_symbols(),
                           default=sensor.SensorSpec.material_symbol,
                           help="active sensor material")
        group.add_argument("--thickness", type=float, default=sensor.SensorSpec.thickness,
                           help="sensor thickness in cm")
        group.add_argument("--diffusion_sigma", type=float, default=40.,
                           help="diffusion sigma in um / cm^1/2")
        group.add_argument("--fano_factor", type=float, default=None,
                           help="fano factor, overriding the tabulated value if specified")

    @staticmethod
    def add_readout_options(parser: argparse.ArgumentParser) -> None:
        """Add an option group for the readout properties.
        """
        group = parser.add_argument_group("readout", "Redout configuration")
        group.add_argument("--layout", type=str, choices=hexagon.HexagonalLayout.values(),
                           default=hexagon.HexagonalGridSpec.layout,
                           help="chip layout")
        group.add_argument("--num_cols", type=int, default=hexagon.HexagonalGridSpec.num_cols,
                           help="number of colums in the readout chip")
        group.add_argument("--num_rows", type=int, default=hexagon.HexagonalGridSpec.num_rows,
                           help="number of rows in the readout chip")
        group.add_argument("--pitch", type=float, default=hexagon.HexagonalGridSpec.pitch,
                           help="pitch of the readout chip in cm")
        # modes = [item.value for item in HexagonalReadoutMode]
        # group.add_argument("--readoutmode", type=str, choices=modes, default="RECTANGULAR",
        #     help="readout mode")
        # group.add_argument("--padding", type=int, nargs=4, default=(2, 2, 2, 2),
        #     help="padding on the four sides of the ROT")
        # group.add_argument("--noise", type=float, default=20.,
        #     help="equivalent noise charge rms in electrons")
        # group.add_argument("--gain", type=float, default=1.,
        #     help="conversion factors between electron equivalent and ADC counts")
        # group.add_argument("--offset", type=int, default=0,
        #     help="optional signal offset in ADC counts")
        # group.add_argument("--trgthreshold", type=float, default=500.,
        #     help="trigger threshold in electron equivalent")
        # group.add_argument("--zsupthreshold", type=int, default=0,
        #     help="zero-suppression threshold in ADC counts")

    @staticmethod
    def filter_namespace(ns: argparse.Namespace, *keys) -> dict:
        """Filter the command-line arguments in a namespace.

        Arguments
        ---------
        ns : argparse.Namespace
            The namespace containing the command-line arguments.

        keys : str
            The keys to filter for.

        Returns
        -------
        dict
            A dictionary containing only the requested keys in the original namespace.
        """
        return {key: getattr(ns, key) for key in keys}

    @staticmethod
    def sensor_from_namespace(ns: argparse.Namespace) -> sensor.Sensor:
        """Create a Sensor object from the command-line arguments.

        Arguments
        ---------
        ns : argparse.Namespace
            The namespace containing the command-line arguments.

        Returns
        -------
        sensor.Sensor
            The sensor object.
        """
        keys = ("material_symbol", "thickness", "diffusion_sigma", "fano_factor")
        return sensor.Sensor(**CliArgumentParser.filter_namespace(ns, *keys))


    def run(self) -> None:
        """Run the actual command tied to the specific options.
        """
        # Parse the command-line arguments. We keep track of the both the namespace
        # object and the corresponding dictionary of command-line arguments.
        # Each sub-command in the main argument parser is tied to a specific function
        # that is accessed through the 'runner' attribute in the namespace.
        ns = self.parse_args()
        kwargs = vars(ns)
        runner = kwargs.pop("runner")
        # Simulate?
        if runner == tasks.simulate:
            src = source.source_factory(**kwargs)
            sensor = self.sensor_from_namespace(ns)
            runner(src, sensor)


def main() -> None:
    """Main entry point.
    """
    start_message()
    CliArgumentParser().run()


if __name__ == "__main__":
    main()
