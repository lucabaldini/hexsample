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

"""System-wide facilities.
"""

import pathlib
import subprocess

from ._version import __version__ as __base_version__
from .logging_ import logger


def _git_suffix() -> str:
    """If we are in a git repo, we want to add the necessary information to the
    version string.

    This will return something along the lines of ``+gf0f18e6.dirty``.
    """
    # pylint: disable=broad-except
    kwargs = dict(cwd=pathlib.Path(__file__).parent, stderr=subprocess.DEVNULL)
    try:
        # Retrieve the git short sha to be appended to the base version string.
        args = ["git", "rev-parse", "--short", "HEAD"]
        sha = subprocess.check_output(args, **kwargs).decode().strip()
        suffix = f"+g{sha}"
        # If we have uncommitted changes, append a `.dirty` to the version suffix.
        args = ["git", "diff", "--quiet"]
        if subprocess.call(args, stdout=subprocess.DEVNULL, **kwargs) != 0:
            suffix = f"{suffix}.dirty"
        return suffix
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


__version__ = f"{__base_version__}{_git_suffix()}"


# Basic package structure.
HEXSAMPLE_ROOT = pathlib.Path(__file__).parent
HEXSAMPLE_BASE = HEXSAMPLE_ROOT.parent
HEXSAMPLE_DOCS = HEXSAMPLE_BASE / "docs"
HEXSAMPLE_DOCS_FIGURES = HEXSAMPLE_DOCS / "figures"
HEXSAMPLE_DOCS_STATIC = HEXSAMPLE_DOCS / "_static"
HEXSAMPLE_TEST = HEXSAMPLE_BASE / "tests"
HEXSAMPLE_TEST_DATA = HEXSAMPLE_TEST / "data"
HEXSAMPLE_BIN = HEXSAMPLE_ROOT / "bin"

# Path to the Python module containing the version information.
HEXSAMPLE_VERSION_FILE_PATH = HEXSAMPLE_ROOT / "_version.py"

# Path to the release notes.
HEXSAMPLE_RELEASE_NOTES_PATH = HEXSAMPLE_DOCS / "release.rst"

# Make room for the output data.
HEXSAMPLE_DATA = pathlib.Path.home() / "hexsampledata"
if not HEXSAMPLE_DATA.exists():
    logger.info(f"Creating data folder {HEXSAMPLE_DATA}...")
    pathlib.Path.mkdir(HEXSAMPLE_DATA)
