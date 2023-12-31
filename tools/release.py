#!/usr/bin/env python
#
# Copyright (C) 2022--2023 luca.baldini@pi.infn.it
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

"""Simple versioning tool.
"""

from enum import Enum
import time
from typing import Tuple

from loguru import logger

from hexsample import __pkgname__, __version__, __tagdate__
from hexsample import HEXSAMPLE_VERSION_FILE_PATH, HEXSAMPLE_RELEASE_NOTES_PATH
from hexsample.shell import cmd


_BUILD_DATE_FORMAT = '%a, %d %b %Y %H:%M:%S %z'

class BumpMode(Enum):

    """Enum class expressing the possible version bumps.
    """

    MAJOR : str = 'major'
    MINOR : str = 'minor'
    MICRO : str = 'micro'


_BUMP_MODES = tuple([mode.value for mode in BumpMode])



def latch_date() -> str:
    """Latch the current date and time
    """
    return time.strftime(_BUILD_DATE_FORMAT)

def build_version_string(major : int, minor : int, micro : int) -> str:
    """Build a version string from the macro, minor and micro fields.

    Arguments
    ---------
    major : int
        The major field of the version string.

    minor : int
        The minor field of the version string.

    micro : int
        The micro field of the version string.
    """
    return f'{major}.{minor}.{micro}'

def parse_version_string(version_string : str) -> Tuple[int, int, int]:
    """Parse a version string.

    Note we are not doing anything fancier than sticking to a simple
    major.minor.micro versioning system, with no alpha, beta or release candidates.

    Arguments
    ---------
    version_string : str
        The version string.
    """
    return [int(item) for item in version_string.split('.')]

def bump_version_string(version_string : str, mode : BumpMode) -> str:
    """Bump a version string with a given bump mode.
    """
    major, minor, micro = parse_version_string(version_string)
    if mode == BumpMode.MAJOR:
        return build_version_string(major + 1, 0, 0)
    if mode == BumpMode.MINOR:
        return build_version_string(major, minor + 1, 0)
    if mode == BumpMode.MICRO:
        return build_version_string(major, minor, micro + 1)
    raise RuntimeError(f'Unknown bump mode {mode}')

def write_version_file(version_string : str, tag_date : str, dry_run : bool = False) -> None:
    """Write the version string and tag date to the unique version file in the package.
    """
    logger.info(f'Writing version info to {HEXSAMPLE_VERSION_FILE_PATH}...')
    logger.info(f'Version string: {version_string}')
    logger.info(f'Tag date: {tag_date}')
    if dry_run:
        logger.info('Dry run, just kidding')
        return
    with open(HEXSAMPLE_VERSION_FILE_PATH, 'w') as version_file:
        version_file.write(f'__version__ = \'{version_string}\'\n')
        version_file.write(f'__tagdate__ = \'{tag_date}\'\n')
    logger.info('Done.')

def update_release_notes(version_string : str, tag_date : str, dry_run : bool = False) -> None:
    """ Write the new tag and build date on top of the release notes.
    """
    title = '.. _release_notes:\n\nRelease notes\n=============\n\n'
    logger.info(f'Reading in {HEXSAMPLE_RELEASE_NOTES_PATH}...')
    with open(HEXSAMPLE_RELEASE_NOTES_PATH) as input_file:
        notes = input_file.read().strip('\n').strip(title)
    logger.info(f'Writing out {HEXSAMPLE_RELEASE_NOTES_PATH}...')
    if dry_run:
        logger.info('Dry run, just kidding')
        return
    with open(HEXSAMPLE_RELEASE_NOTES_PATH, 'w') as release_notes:
        release_notes.writelines(title)
        release_notes.writelines(f'\n*{__pkgname__} ({version_string}) - {tag_date}*\n\n')
        release_notes.writelines(notes)
    logger.info('Done.')

def tag_package(mode : BumpMode, dry_run : bool = False) -> None:
    """Tag the package.
    """
    cmd('git pull', dry_run)
    cmd('git status', dry_run)
    version_string = bump_version_string(__version__, mode)
    tag_date = latch_date()
    write_version_file(version_string, tag_date, dry_run)
    update_release_notes(version_string, tag_date, dry_run)
    message = f'Prepare for tag {version_string}.'
    cmd(f'git commit -a -m "{message}"', dry_run)
    cmd('git push', dry_run)
    message = f'Tagging version {version_string}'
    cmd(f'git tag -a {version_string} -m "{message}"', dry_run)
    cmd('git push --tags', dry_run)
    cmd('git status', dry_run)



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('mode', type=str, choices=_BUMP_MODES,
        help='the release bump mode')
    parser.add_argument('--dryrun', action='store_true', default=False,
        help='dry run')
    args = parser.parse_args()
    tag_package(BumpMode(args.mode), args.dryrun)
