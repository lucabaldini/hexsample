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

from hexsample import __name__ as __package_name__
from hexsample import __version__, hexagon, readout, sensor, source, tasks


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
        self.add_num_events(simulate, default=1000, intent="generated")
        self.add_output_file(simulate, default="simulation_output.h5")
        self.add_random_seed(simulate)
        self.add_source_options(simulate)
        self.add_sensor_options(simulate)
        self.add_readout_options(simulate)
        simulate.set_defaults(runner=tasks.simulate)

        # Run the event reconstruction?
        #recon = subparsers.add_parser("recon",
        #    help="run the event reconstruction",
        #    formatter_class=self._FORMATTER_CLASS)

        # Run the single-event display?
        #display = subparsers.add_parser("display",
        #    help="run the single-event display",
        #    formatter_class=self._FORMATTER_CLASS)

    @staticmethod
    def add_input_file(parser: argparse.ArgumentParser) -> None:
        """Add an option for the input file.
        """
        parser.add_argument("input_file", type=str,
            help="path to the input file")

    @staticmethod
    def add_num_events(parser: argparse.ArgumentParser, default: int,
                       intent: str = "generated") -> None:
        """Add an option for the number of events.
        """
        parser.add_argument("--num_events", "-n", type=int, default=default,
            help=f"number of events to be {intent}")

    @staticmethod
    def add_output_file(parser: argparse.ArgumentParser, default: str) -> None:
        """Add an option for the output file.

        Note that we cast the default to a string---this prevents having
        pathlib.Path instances around, which would then needed to be handled
        properly in specific places (such as adding metadata to the output HDF5
        file headers).
        """
        parser.add_argument("--output_file", "-o", type=str, default=str(default),
            help="path to the output file")

    @staticmethod
    def add_random_seed(parser: argparse.ArgumentParser) -> None:
        """Add an option for the random seed of a simulation.
        """
        parser.add_argument("--random_seed", "-s", type=int, default=None,
            help="random seed for the simulation")

    @staticmethod
    def add_suffix(parser: argparse.ArgumentParser, default: str) -> None:
        """Add an option for the output suffix.
        """
        parser.add_argument("--suffix", type=str, default=default,
            help="suffix for the output file")

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
        group.add_argument(f"--{source.SpectrumProxy.key()}", type=str,
                           choices=source.SpectrumProxy.choices(),
                           default=source.SpectrumProxy.default(),
                           help="spectrum of the X-ray source")
        group.add_argument("--energy", type=float, default=source.Line.energy,
                           help="line energy in eV")
        group.add_argument("--element", type=str, default=source.LineForest.element,
                           help="element generating the line forest")
        group.add_argument("--initial_level", type=str, default=source.LineForest.initial_level,
                           help="initial level for the line forest")
        # ... morphological part...
        group.add_argument(f"--{source.BeamProxy.key()}", type=str,
                           choices=source.BeamProxy.choices(),
                           default=source.BeamProxy.default(),
                           help="beam shape of the X-ray source")
        group.add_argument("--x0", type=float, default=source.PointBeam.x0,
                           help="x-coordinate of the beam centroid in cm")
        group.add_argument("--y0", type=float, default=source.PointBeam.y0,
                           help="y-coordinate of the beam centroid in cm")
        group.add_argument("--radius", type=float, default=source.DiskBeam.radius,
                           help="radius of the disk beam in cm")
        group.add_argument("--sigma", type=float, default=source.GaussianBeam.sigma,
                           help="standard deviation of the gaussian beam in cm")
        # ... and overall rate.
        group.add_argument("--rate", type=float, default=source.Source.rate,
                           help="source rate in photons/s")

    @staticmethod
    def add_sensor_options(parser: argparse.ArgumentParser) -> None:
        """Add an option group for the sensor properties.
        """
        group = parser.add_argument_group("sensor", "Sensor properties")
        group.add_argument("--material_symbol", type=str, choices=sensor.material_symbols(),
                           default=sensor.Sensor.material_symbol,
                           help="active sensor material")
        group.add_argument("--thickness", type=float, default=sensor.Sensor.thickness,
                           help="sensor thickness in cm")
        group.add_argument("--diffusion_sigma", type=float, default=sensor.Sensor.diffusion_sigma,
                           help="diffusion sigma in um / cm^1/2")
        group.add_argument("--fano_factor", type=float, default=None,
                           help="fano factor, overriding the tabulated value if specified")

    @staticmethod
    def add_readout_options(parser: argparse.ArgumentParser) -> None:
        """Add an option group for the readout properties.
        """
        group = parser.add_argument_group("readout", "Redout configuration")
        group.add_argument("--layout", type=str, choices=hexagon.HexagonalLayout.values(),
                           default=hexagon.HexagonalGrid.layout,
                           help="chip layout")
        group.add_argument("--num_cols", type=int, default=hexagon.HexagonalGrid.num_cols,
                           help="number of colums in the readout chip")
        group.add_argument("--num_rows", type=int, default=hexagon.HexagonalGrid.num_rows,
                           help="number of rows in the readout chip")
        group.add_argument("--pitch", type=float, default=hexagon.HexagonalGrid.pitch,
                           help="pitch of the readout chip in cm")
        group.add_argument("--enc", type=float, default=readout.HexagonalReadoutBase.enc,
                           help="equivalent noise charge in electrons")
        group.add_argument("--gain", type=float, default=readout.HexagonalReadoutBase.gain,
                           help="conversion factor between electron equivalent and ADC counts")
        group.add_argument("--trg_threshold", type=float, default=readout.HexagonalReadoutBase.trg_threshold,
                           help="trigger threshold in electron equivalent")
        group.add_argument("--zero_sup_threshold", type=int, default=readout.HexagonalReadoutBase.zero_sup_threshold,
                           help="zero suppression threshold in ADC counts")
        group.add_argument(f"--{readout.ReadoutProxy.key()}", type=str,
                           choices=readout.ReadoutProxy.choices(),
                           default=readout.ReadoutProxy.default(),
                           help="chip readout mode")
        #group.add_argument("--padding", type=int, nargs=4, default=(2, 2, 2, 2),
        #                   help="padding on the four sides of the ROT")

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
            _source = source.Source.from_kwargs(**kwargs)
            _sensor = sensor.Sensor.from_kwargs(**kwargs)
            _readout = readout.ReadoutProxy.from_kwargs(**kwargs)
            runner(_source, _sensor, _readout)


def main() -> None:
    """Main entry point.
    """
    start_message()
    CliArgumentParser().run()


if __name__ == "__main__":
    main()
