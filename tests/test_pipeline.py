# Copyright (C) 2023 luca.baldini@pi.infn.it
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

"""Test suite for hexsample.pipeline
"""

from hexsample import HEXSAMPLE_DATA
import hexsample.pipeline as pipeline


def test_parsers():
    """Test the relevant ArgumentParser objects.
    """
    print(pipeline.update_args(pipeline.HXSIM_ARGPARSER))
    print(pipeline.update_args(pipeline.HXRECON_ARGPARSER, ['infile']))
    print(pipeline.update_args(pipeline.HXRECON_ARGPARSER, ['infile'], infile='test_file'))

def test_pipeline():
    """Test generating and reconstructing files.
    """
    for thickness in (0.02, 0.03, 0.04):
        output_file_path = HEXSAMPLE_DATA / f'hxsim_{thickness}.h5'
        file_path = pipeline.hxsim(numevents=100, thickness=thickness, outfile=output_file_path)
        for nneighbors in (3, 4):
            pipeline.hxrecon(infile=file_path, suffix=f'recon_nn{nneighbors}')
