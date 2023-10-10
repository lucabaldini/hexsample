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

"""System utilities.
"""

import subprocess

from loguru import logger


def cmd(command : str, dry_run : bool = False) -> int:
    """ Exec a system command.

    Arguments
    ---------
    command : str
        The command to be executed.

    dry_run : bool
        Boolean flag for testing purposes---if true the command is not executed.

    Returns
    -------
    error_code : int
        The error code from the execution of the command.
    """
    logger.info(f'About to execute "{command}"...')
    # If this is a dry run, do nothing and return zero.
    if dry_run:
        logger.info('Just kidding (dry run).')
        return 0
    # Run the actual command, collect the output and return the error code.
    kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    with subprocess.Popen(command, **kwargs) as process:
        error_code = process.wait()
        print(process.stdout.read().strip(b'\n').decode())
        if error_code:
            logger.error('Command returned status code %d.', error_code)
            print(process.stderr.read().decode().strip('\n'))
        return error_code
