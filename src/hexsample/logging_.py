# Copyright (C) 2025 the hexsample team.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Logging facilities.
"""

import sys

from loguru import logger

from . import __name__ as __package_name__

# Initial logger configuration: disable all logging removing all sinks.
logger.remove()
# Disable log emission from this namespace by default.
logger.disable(__package_name__)


def setup_logger(level: str = "DEBUG") -> int:
    """Minimal configuration facility for the loguru logger.

    A few remarks about the loguru internals. In order to keep the API clean, the
    author of the library made the deliberate decision not to allow to change handlers,
    so that the preferred way to change the logger configuration is to remove all the
    existing handlers and start from scratch---this is exactly what we are doing here.

    Also note that whenever you add a new handler, you get back an ID that can be used
    to remove the handler later on. The default handler (which we get rid of at the
    first call to this function) is guaranteed to have ID 0.

    Arguments
    ---------
    level : str
        The minimum logging level to be used by the logger. Defaults to "DEBUG".
        Other possible values are "INFO", "WARNING", "ERROR", and "CRITICAL".

    Returns
    -------
    int
        The ID of the handler that was added.
    """
    # Enable logging for this namespace.
    logger.enable(__package_name__)
    # Remove all existing handlers.
    logger.remove()
    # Create a plain, terminal-based logger.
    fmt = ">>> <level>[{level}] {message}</level>"
    return logger.add(sys.stderr, level=level, colorize=True, format=fmt)


def start_file_logging(file_path: str, level: str = "DEBUG") -> int:
    """Add a new file-based handler to the logger.

    Arguments
    ---------
    file_path : str
        The path to the file where the log messages should be written.

    level : str
        The minimum logging level to be used by the logger. Defaults to "DEBUG".

    Returns
    -------
    int
        The ID of the handler that was added.
    """
    logger.info(f"Directing logger output to {file_path} (level={level})...")
    return logger.add(file_path, level=level)
