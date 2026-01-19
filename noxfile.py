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

import pathlib
import shutil

import nox

_ROOT_DIR_PATH = pathlib.Path(__file__).parent
_DOCS_DIR_PATH = _ROOT_DIR_PATH / "docs"

_LINT_DIRS = ("src", "tests", "tools")
_TEST_DIRS = ("tests", )
_CACHE_DIRS = (".nox", ".ruff_cache", ".pylint_cache", ".pytest_cache")
_DOCS_ARTIFACTS_DIRS = ("_build", "auto_examples")

# Reuse existing virtualenvs by default.
nox.options.reuse_existing_virtualenvs = True


def _rm(file_path: pathlib.Path, session: nox.Session) -> None:
    """Remove a file or directory at the given path.
    """
    if not file_path.exists():
        return
    if file_path.is_dir():
        session.log(f"Removing folder {file_path}...")
        shutil.rmtree(file_path)
    elif file_path.is_file():
        session.log(f"Removing file {file_path}...")
        file_path.unlink()


@nox.session(venv_backend="none")
def clean(session: nox.Session) -> None:
    """Cleanup build artifacts and caches.
    """
    session.log("Cleaning up build artifacts and caches...")
    # Directories or patterns to remove
    patterns = ("__pycache__", )
    # Loop through the patterns and remove matching files/directories...
    for pattern in patterns:
        for _path in _ROOT_DIR_PATH.rglob(pattern):
            if any(folder_name in _path.parts for folder_name in _CACHE_DIRS):
                continue
            _rm(_path, session)
    # Cleanup the docs.
    session.log("Cleaning up documentation build artifacts...")
    for folder_name in _DOCS_ARTIFACTS_DIRS:
        _rm(_DOCS_DIR_PATH / folder_name, session)


@nox.session(venv_backend="none")
def cleanall(session: nox.Session) -> None:
    """Cleanup literally anything that is not in the repo.
    """
    session.notify("clean")
    for folder_name in _CACHE_DIRS:
        _rm(_ROOT_DIR_PATH / folder_name, session)


@nox.session(venv_backend="none")
def doc(session: nox.Session) -> None:
    """Build the HTML docs.

    Note this is a nox session with no virtual environment, based on the assumption
    that it is not very interesting to build the documentation with different
    versions of Python or the associated environment, since the final thing will
    be created remotely anyway. (This also illustrates the use of the nox.session
    decorator called with arguments.)
    """
    source_dir = _DOCS_DIR_PATH
    output_dir = _DOCS_DIR_PATH / "_build" / "html"
    session.run("sphinx-build", "-b", "html", source_dir, output_dir, *session.posargs)


@nox.session
def ruff(session: nox.Session) -> None:
    """Run ruff.
    """
    session.install("ruff")
    session.install(".[dev]")
    session.run("ruff", "check", *session.posargs)


@nox.session
def pylint(session: nox.Session) -> None:
    """Run pylint.
    """
    session.install("pylint")
    session.install(".[dev]")
    session.run("pylint", *_LINT_DIRS, *session.posargs)


@nox.session
def test(session: nox.Session) -> None:
    """Run the unit tests.
    """
    session.install("pytest")
    session.install(".[dev]")
    session.run("pytest", *_TEST_DIRS, *session.posargs)