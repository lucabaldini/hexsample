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
from hexsample import (
    __version__,
    caldb,
    calibration,
    hexagon,
    logging_,
    pdf,
    pipeline,
    readout,
    roi,
    sensor,
    source,
    tasks,
    xpol,
)


def start_message() -> None:
    """Print the start message.
    """
    msg = f"""
    This is {__package_name__} version {__version__}.

    Copyright (C) 2023--2026, the {__package_name__} team.

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


# pylint: disable=too-many-public-methods
class CliArgumentParser(argparse.ArgumentParser):

    """Application-wide argument parser.
    """

    _DESCRIPTION = None
    _EPILOG = None
    _FORMATTER_CLASS = _Formatter

    def __init__(self) -> None:
        """Overloaded method.
        """
        # pylint: disable=too-many-statements
        super().__init__(description=self._DESCRIPTION, epilog=self._EPILOG,
                         formatter_class=self._FORMATTER_CLASS)
        subparsers = self.add_subparsers(required=True, help="sub-command help")
        # See https://stackoverflow.com/questions/8757338/
        subparsers._parser_class = argparse.ArgumentParser

        # Run a simulation?
        simulate = subparsers.add_parser("simulate",
            help="run a simulation",
            formatter_class=self._FORMATTER_CLASS)
        self.add_num_events(simulate, default=tasks.SimulationDefaults.num_events,
                            intent="generated")
        self.add_output_file(simulate, default=tasks.SimulationDefaults.output_file_path)
        self.add_random_seed(simulate, default=tasks.SimulationDefaults.random_seed)
        self.add_logging_level(simulate)
        self.add_source_options(simulate)
        self.add_sensor_options(simulate)
        self.add_readout_options(simulate)
        simulate.set_defaults(runner=pipeline.simulate)

        # Run the event reconstruction?
        recon = subparsers.add_parser("reconstruct",
            help="run the event reconstruction",
            formatter_class=self._FORMATTER_CLASS)
        self.add_input_file(recon)
        self.add_suffix(recon, default="recon")
        self.add_logging_level(recon)
        self.add_recon_options(recon)
        recon.set_defaults(runner=pipeline.reconstruct)

        # Run the chip calibration?
        calibrate = subparsers.add_parser("calibgen",
            help="generate detector calibration files",
            formatter_class=self._FORMATTER_CLASS)
        calibrate_subparsers = calibrate.add_subparsers(required=True, help="calibration mode")

        # Dark calibration
        dark = calibrate_subparsers.add_parser("dark", help="calibrate the chip noise and pedestal")
        self.add_input_file(dark)
        self.add_calibrate_dark_options(dark)
        self.add_logging_level(dark)
        dark.set_defaults(runner=pipeline.calibrate_dark)
        # ENC calibration
        enc = calibrate_subparsers.add_parser("enc", help="calibrate the chip ENC")
        self.add_enc_calibration_options(enc)
        self.add_logging_level(enc)
        enc.set_defaults(runner=pipeline.calibrate_enc)
        # Pixel equalization calibration
        equalization = calibrate_subparsers.add_parser("equalization",
                                                    help="calibrate the chip pixel equalization")
        self.add_input_file(equalization)
        self.add_calibrate_equalization_options(equalization)
        self.add_logging_level(equalization)
        equalization.set_defaults(runner=pipeline.calibrate_equalization)
        # Eta function calibration
        eta = calibrate_subparsers.add_parser("eta", help="calibrate the eta function")
        self.add_input_file(eta)
        self.add_calibrate_eta_options(eta)
        self.add_logging_level(eta)
        eta.set_defaults(runner=pipeline.calibrate_eta)
        # Gain calibration
        gain = calibrate_subparsers.add_parser("gain", help="calibrate the chip gain")
        self.add_gain_calibration_options(gain)
        self.add_logging_level(gain)
        gain.set_defaults(runner=pipeline.calibrate_gain)
        # Noise calibration
        noise = calibrate_subparsers.add_parser("noise", help="calibrate the chip noise")
        self.add_input_file(noise)
        self.add_logging_level(noise)
        noise.set_defaults(runner=pipeline.calibrate_noise)
        # Position reconstruction calibration
        position = calibrate_subparsers.add_parser("position",
                                    help="calibrate the position reconstruction algorithms")
        self.add_input_file(position)
        self.add_calibrate_position_options(position)
        self.add_logging_level(position)
        position.set_defaults(runner=pipeline.calibrate_position)
        # Synthesize calibration files
        synthesize = calibrate_subparsers.add_parser("synthesize",
            help="generate synthetic calibration files")
        self.add_synthesize_calibration_file_options(synthesize)
        self.add_logging_level(synthesize)
        synthesize.set_defaults(runner=pipeline.synthesize_calibration_file)

        # Fit a spectrum to generate a PDF for the gain calibration?
        calibspec = subparsers.add_parser("calibspec",
            help="generate a pdf from a spectrum to use in the gain calibration",
            formatter_class=self._FORMATTER_CLASS)
        self.add_input_file(calibspec)
        self.add_logging_level(calibspec)
        calibspec.set_defaults(runner=pipeline.calibspec)

        # Inspect a calibration matrix?
        calibview = subparsers.add_parser("calibview",
            help="inspect a calibration matrix",
            formatter_class=self._FORMATTER_CLASS)
        self.add_calibview_options(calibview)
        self.add_logging_level(calibview)
        calibview.set_defaults(runner=pipeline.calibview)

        # Run the single-event display?
        display = subparsers.add_parser("display",
            help="run the single-event display",
            formatter_class=self._FORMATTER_CLASS)
        self.add_input_file(display)
        self.add_display_options(display)
        self.add_logging_level(display)
        display.set_defaults(runner=pipeline.display)

        # Run the quicklook?
        quicklook = subparsers.add_parser("quicklook",
            help="run a quick-look analysis of a recon file",
            formatter_class=self._FORMATTER_CLASS)
        self.add_input_file(quicklook)
        self.add_logging_level(quicklook)
        quicklook.set_defaults(runner=pipeline.quicklook)

        # Convert a .mdat3 file to a HDF5 digi file?
        convert = subparsers.add_parser("convert",
            help="convert a .mdat3 file to a HDF5 digi file",
            formatter_class=self._FORMATTER_CLASS)
        self.add_input_file(convert)
        self.add_num_events(convert, default=None, intent="converted")
        self.add_logging_level(convert)
        convert.set_defaults(runner=pipeline.mdat3_to_digi)

    @staticmethod
    def add_input_file(parser: argparse.ArgumentParser) -> None:
        """Add an option for the input file.
        """
        parser.add_argument("input_file", type=str,
            help="path to the input file")

    @staticmethod
    def add_logging_level(parser: argparse.ArgumentParser) -> None:
        """Add an option for the input file.
        """
        parser.add_argument("--logging_level", type=str, choices=logging_.logging_levels(),
                            default="INFO",
                            help="logging level")

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
    def add_random_seed(parser: argparse.ArgumentParser, default: int) -> None:
        """Add an option for the random seed of a simulation.
        """
        parser.add_argument("--random_seed", "-s", type=int, default=default,
                            help="random seed for the simulation")

    @staticmethod
    def add_suffix(parser: argparse.ArgumentParser, default: str) -> None:
        """Add an option for the output suffix.
        """
        parser.add_argument("--suffix", type=str, default=default,
                            help="suffix for the output file")

    @staticmethod
    def add_num_bins(parser: argparse.ArgumentParser, default: int) -> None:
        """Add an option for the number of bins to be used in different calibrations.
        """
        parser.add_argument("--num_bins", type=int, default=default,
                            help="number of bins to be used in the calibration")

    @staticmethod
    def add_zero_sup_threshold(parser: argparse.ArgumentParser, default: float) -> None:
        """Add an option for the zero-suppression threshold.
        """
        parser.add_argument("--zero_sup_threshold", type=float, default=default,
                            help="zero-suppression threshold expressed as sigma of noise")

    @staticmethod
    def add_cal_enc_file(parser: argparse.ArgumentParser, default: str,
                         required: bool = False) -> None:
        """Add an option for the ENC calibration file.
        """
        parser.add_argument("--enc", type=caldb.CalDB.open_enc, default=default,
                            required=required, help="path to a file containing the ENC " \
                            "calibration data or name of a calibration file inside the " \
                            "caldb/enc folder.")

    @staticmethod
    def add_cal_noise_file(parser: argparse.ArgumentParser, default: str,
                           required: bool = False) -> None:
        """Add an option for the noise calibration file.
        """
        parser.add_argument("--noise", type=caldb.CalDB.open_noise, default=default,
                            required=required, help="path to a file containing the noise " \
                            "calibration data or name of a calibration file inside the " \
                            "caldb/noise folder.")

    @staticmethod
    def add_cal_pedestal_file(parser: argparse.ArgumentParser, default: str,
                              required: bool = False) -> None:
        """Add an option for the pedestal calibration file.
        """
        parser.add_argument("--pedestal", type=caldb.CalDB.open_pedestal, default=default,
                            required=required, help="path to a file containing the pedestal " \
                            "calibration data or name of a calibration file inside the " \
                            "caldb/pedestal folder.")

    @staticmethod
    def add_cal_equalization_file(parser: argparse.ArgumentParser, default: str,
                                  required: bool = False) -> None:
        """Add an option for the equalization calibration file.
        """
        parser.add_argument("--equalization", type=caldb.CalDB.open_equalization, default=default,
                            required=required, help="path to a file containing the equalization " \
                            "calibration data or name of a calibration file inside the " \
                            "caldb/equalization folder.")

    @staticmethod
    def add_cal_gain_file(parser: argparse.ArgumentParser, default: str,
                          required: bool = False) -> None:
        """Add an option for the gain calibration file.
        """
        parser.add_argument("--gain", type=caldb.CalDB.open_gain, default=default,
                            required=required, help="path to a file containing the gain " \
                            "calibration data or name of a calibration file inside the " \
                            "caldb/gain folder.")

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
        group.add_argument("--x0", type=float, default=source.AbstractBeam.x0,
                           help="x-coordinate of the beam centroid in cm")
        group.add_argument("--y0", type=float, default=source.AbstractBeam.y0,
                           help="y-coordinate of the beam centroid in cm")
        group.add_argument("--radius", type=float, default=source.DiskBeam.radius,
                           help="radius of the disk beam in cm")
        group.add_argument("--side", type=float, default=source.SquareBeam.side,
                           help="side of the square beam in cm")
        group.add_argument("--width", type=float, default=source.RectangleBeam.width,
                           help="width of the rectangle beam in cm")
        group.add_argument("--height", type=float, default=source.RectangleBeam.height,
                           help="height of the rectangle beam in cm")
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
        CliArgumentParser.add_cal_enc_file(
            group, default="sim_xpol3_enc-20_uniform_v001", required=False)
        CliArgumentParser.add_cal_pedestal_file(
            group, default="sim_xpol3_pedestal-1000_uniform_v001", required=False)
        CliArgumentParser.add_cal_gain_file(
            group, default="sim_xpol3_gain-1_uniform_v001", required=False)
        group.add_argument("--layout", type=str, choices=hexagon.HexagonalLayout.values(),
                           default=hexagon.HexagonalGrid.layout,
                           help="chip layout")
        group.add_argument("--num_cols", type=int, default=hexagon.HexagonalGrid.num_cols,
                           help="number of columns in the readout chip")
        group.add_argument("--num_rows", type=int, default=hexagon.HexagonalGrid.num_rows,
                           help="number of rows in the readout chip")
        group.add_argument("--pitch", type=float, default=hexagon.HexagonalGrid.pitch,
                           help="pitch of the readout chip in cm")
        group.add_argument(f"--{readout.ReadoutProxy.key()}", type=str,
                           choices=readout.ReadoutProxy.choices(),
                           default=readout.ReadoutProxy.default(),
                           help="chip readout mode")
        # Note this one reqquires 4 int arguments, and we do need to convert the
        # iterable to an actual roi.Padding instance after the parse_args() call.
        group.add_argument("--padding", type=int, nargs=4,
                           default=readout.HexagonalReadoutRectangular.padding,
                           help="padding on the four sides of the ROT")
        group.add_argument("--trg_threshold", type=float,
                           default=readout.HexagonalReadoutBase.trg_threshold,
                           help="trigger threshold in electron equivalent")
        CliArgumentParser.add_zero_sup_threshold(group,
                           default=readout.HexagonalReadoutBase.zero_sup_threshold)

    def add_recon_options(self, parser: argparse.ArgumentParser) -> None:
        """Add an option group for the reconstruction properties.
        """
        defaults = tasks.ReconstructionDefaults
        group = parser.add_argument_group("reconstruction", "Reconstruction configuration")
        CliArgumentParser.add_zero_sup_threshold(group, default=defaults.zero_sup_threshold)
        group.add_argument("--num_neighbors", type=int, default=2,
                           help="number of neighbors to be considered (0--6)")
        group.add_argument("--max_neighbors", type=int, default=-1,
                           help="maximum number of neighbors to be considered")
        group.add_argument("--pos_recon_algorithm", choices=["centroid", "eta", "mle"],
                           type=str, default="centroid", help="How to reconstruct position")
        CliArgumentParser.add_cal_noise_file(group, default=None, required=True)
        CliArgumentParser.add_cal_pedestal_file(group, default=None, required=True)
        CliArgumentParser.add_cal_equalization_file(group, default=None, required=True)
        group.add_argument("--eta_2pix_rad_sigma", default=defaults.eta_2pix_rad_sigma, type=float,
                           help="probit function sigma parameter for two pixel" \
                           "events eta reconstruction")
        group.add_argument("--eta_2pix_rad_pivot", default=defaults.eta_2pix_rad_pivot, type=float,
                           help="transition value from linear (0 to pivot) to probit (> pivot) " \
                           "for two pixel events eta reconstruction")
        group.add_argument("--eta_3pix_rad_offset", default=defaults.eta_3pix_rad_offset,
                           type=float, help="probit function offset parameter for three pixel" \
                           " events radial component eta reconstruction")
        group.add_argument("--eta_3pix_rad_sigma", default=defaults.eta_3pix_rad_sigma, type=float,
                           help="probit function sigma parameter for three pixel " \
                           "events radial component eta reconstruction")
        group.add_argument("--eta_3pix_rad_pivot", default=defaults.eta_3pix_rad_pivot, type=float,
                           help="transition value from linear (0 to pivot) to probit (> pivot) " \
                           "for three pixel events radial component eta reconstruction")
        group.add_argument("--eta_3pix_theta_sigma", default=defaults.eta_3pix_theta_sigma,
                           type=float, help="probit function sigma parameter for three pixel " \
                           "events angular component eta reconstruction")
        group.add_argument("--mle_data", type=caldb.CalDB.open_mle, default=None, required=False,
                           help="path to a file containing the MLE calibration data or name of a " \
                           "calibration file inside the caldb/mle folder.")

    def add_calibrate_dark_options(self, parser: argparse.ArgumentParser) -> None:
        """Add an option group for the dark calibration properties.
        """
        defaults = tasks.CalibrationDarkDefaults
        parser.add_argument("--algorithm", type=str, choices=["welford", "fit"],
                            default=defaults.algorithm,
                            help="algorithm to be used for the dark calibration")
        parser.add_argument("--no_source", action="store_false", dest="has_source",
                            default=defaults.has_source,
                            help="if specified, events are considered to be without source")
        parser.add_argument("--batch_size", type=int,
                            default=defaults.batch_size,
                            help="number of events to be analyzed in a batch for the dark" \
                            " calibration")

    def add_calibrate_equalization_options(self, parser: argparse.ArgumentParser) -> None:
        """Add an option group for the gain calibration properties.
        """
        defaults = tasks.CalibrationEqualizationDefaults
        CliArgumentParser.add_cal_noise_file(parser, default=None, required=True)
        CliArgumentParser.add_cal_pedestal_file(parser, default=None, required=True)
        parser.add_argument("--algorithm", type=str, choices=["relative", "absolute"],
                            default=defaults.algorithm,
                            help="algorithm to be used for the equalization calibration")
        parser.add_argument("--pdf", type=pdf.SpectrumPDF.from_file, default=defaults.pdf,
                            help="path to the spectrum PDF file")
        parser.add_argument("--size", type=int, default=defaults.size,
                            help="length of the square region of the chip to fit simultaneously")
        CliArgumentParser.add_zero_sup_threshold(parser, default=defaults.zero_sup_threshold)

    def add_calibrate_eta_options(self, parser: argparse.ArgumentParser) -> None:
        """Add an option group for the eta function calibration properties.
        """
        defaults = tasks.CalibrationEtaDefaults
        CliArgumentParser.add_cal_noise_file(parser, default=None, required=True)
        CliArgumentParser.add_cal_pedestal_file(parser, default=None, required=True)
        CliArgumentParser.add_cal_equalization_file(parser, default=None, required=True)
        parser.add_argument("--num_bins", type=int,
                            default=defaults.num_bins,
                            help="number of bins to be used in the eta function calibration")
        CliArgumentParser.add_zero_sup_threshold(parser, default=defaults.zero_sup_threshold)

    def add_enc_calibration_options(self, parser: argparse.ArgumentParser) -> None:
        """Add an option group for the ENC calibration properties.
        """
        defaults = tasks.CalibrationEncDefaults
        CliArgumentParser.add_cal_noise_file(parser, default=None, required=True)
        CliArgumentParser.add_cal_gain_file(parser, default=None, required=True)
        parser.add_argument("--output_dir", type=str, default=defaults.output_dir,
                            help="directory where the generated ENC calibration file" \
                            " will be saved")

    def add_gain_calibration_options(self, parser: argparse.ArgumentParser) -> None:
        """Add an option group for the gain calibration properties.
        """
        defaults = tasks.CalibrationGainDefaults
        CliArgumentParser.add_cal_equalization_file(parser, default=None, required=True)
        parser.add_argument("--material_symbol", type=str,
                            choices=sensor.material_symbols(),
                            default=defaults.material_symbol,
                            help="active sensor material")
        parser.add_argument("--output_dir", type=str, default=defaults.output_dir,
                            help="directory where the generated gain calibration file" \
                            " will be saved")

    def add_synthesize_calibration_file_options(self, parser: argparse.ArgumentParser) -> None:
        """Add an option group to generate calibration files.
        """
        defaults = tasks.SynthesizeCalibrationDefaults
        cal_type = calibration.CalibrationType
        parser.add_argument("calibration_type", type=calibration.CalibrationType,
                            choices=[c.value for c in cal_type if c not in (cal_type.POSITION, cal_type.ETA)],
                            help="type of calibration file to be generated")
        parser.add_argument("mean", type=float,
                            help="mean value of the calibration parameter.")
        parser.add_argument("--percent_rms", type=int, default=defaults.percent_rms,
                            help="relative RMS (as percentage of the mean) of the gaussian" \
                            " distribution. A value of 0 generates a uniform distribution.")
        parser.add_argument("--chip_name", type=str, choices=xpol.chip_names(),
                            default=defaults.chip_name,
                            help="XPOL chip name for which the calibration file is generated." \
                            " This parameter is used to determine the size of the calibration" \
                            " matrix.")
        parser.add_argument("--output_dir", type=str, default=defaults.output_dir,
                            help="directory where the generated calibration file will be saved")
        parser.add_argument("--version", type=int, default=defaults.version,
                            help="version number to be included in the generated calibration" \
                            " file name")
        parser.add_argument("--random_seed", type=int, default=defaults.random_seed,
                            help="random seed for the generation of the calibration values")

    def add_calibrate_position_options(self, parser: argparse.ArgumentParser) -> None:
        """Add an option group for the MLE position reconstruction calibration properties.
        """
        defaults = tasks.CalibrationMLEDefaults
        CliArgumentParser.add_cal_noise_file(parser, default=None, required=True)
        CliArgumentParser.add_cal_pedestal_file(parser, default=None, required=True)
        CliArgumentParser.add_cal_equalization_file(parser, default=None, required=True)
        parser.add_argument("--bin_size", type=float, default=defaults.bin_size,
                            help="bin size to be used in the calibration, in units of pixel pitch")

    def add_display_options(self, parser: argparse.ArgumentParser) -> None:
        """Add an option group for the single-event display properties.
        """
        default = tasks.DisplayDefaults
        CliArgumentParser.add_cal_noise_file(parser, default=default.noise_matrix)
        CliArgumentParser.add_cal_pedestal_file(parser, default=default.pedestal_matrix)
        CliArgumentParser.add_cal_equalization_file(parser, default=default.equalization_matrix)

    def add_calibview_options(self, parser: argparse.ArgumentParser) -> None:
        """Add an option group for the calibration view properties.
        """
        defaults = tasks.CalibviewDefaults
        parser.add_argument("matrix", type=calibration.CalibrationMatrix.from_hdf5,
                            help="path to a calibration matrix to be analyzed")
        parser.add_argument("--mc_matrix", type=calibration.CalibrationMatrix.from_hdf5,
                            default=defaults.mc_matrix,
                            help="path to a calibration matrix containing the Monte Carlo truth" \
                            " matrix to be compared with the main matrix")
        parser.add_argument("--min_hits", type=int, default=defaults.min_hits,
                            help="minimum number of entries for a pixel to be included in" \
                            " the statistics")
        parser.add_argument("--rel_error", type=float, default=defaults.rel_error,
                            help="maximum relative error threshold for a pixel to be included"
                            " in the statistics")
        parser.add_argument("--lower_quantile", type=float, default=defaults.lower_quantile,
                            help="lower quantile for a pixel to be included in the statistics")
        parser.add_argument("--upper_quantile", type=float, default=defaults.upper_quantile,
                            help="upper quantile for a pixel to be included in the statistics")

    def run(self) -> None:
        """Run the actual command tied to the specific options.
        """
        # Parse the command-line arguments. We keep track of the both the namespace
        # object and the corresponding dictionary of command-line arguments.
        # Each sub-command in the main argument parser is tied to a specific function
        # that is accessed through the 'runner' attribute in the namespace.
        ns = self.parse_args()
        # Convert padding to a roi.Padding instance.
        if "padding" in ns and not isinstance(ns.padding, roi.Padding):
            ns.padding = roi.Padding(*ns.padding)
        kwargs = vars(ns)
        # Setup logging.
        logging_.setup_logger(kwargs.pop("logging_level"))
        # Call the appropriate runner function.
        runner = kwargs.pop("runner")
        return runner(**kwargs)


def main() -> None:
    """Main entry point.
    """
    start_message()
    CliArgumentParser().run()


if __name__ == "__main__":
    main()
