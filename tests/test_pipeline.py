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

import pytest

from hexsample import HEXSAMPLE_DATA
import hexsample.pipeline as pipeline


def test_parsers():
    """Test the relevant ArgumentParser objects.
    """
    assert pipeline.required_arguments(pipeline.HXSIM_ARGPARSER) == []
    assert pipeline.required_arguments(pipeline.HXRECON_ARGPARSER) == ['infile']
    assert 'infile' not in pipeline.update_arguments(pipeline.HXRECON_ARGPARSER)
    assert 'infile' in pipeline.update_arguments(pipeline.HXRECON_ARGPARSER, infile='test_file')
    print(pipeline.update_arguments(pipeline.HXSIM_ARGPARSER))

def test_wrong_args():
    """Make sure that, when supplied with wrong parameters, the pipeline applications
    are raising a RuntimeError.
    """
    with pytest.raises(RuntimeError) as excinfo:
        pipeline.hxsim(numevents=100, bogusparam='howdy')
    print(excinfo.value)

def test_pipeline():
    """Test generating and reconstructing files.
    """
    for thickness in (0.02, 0.03, 0.04):
        output_file_path = HEXSAMPLE_DATA / f'hxsim_{thickness}.h5'
        file_path = pipeline.hxsim(numevents=100, thickness=thickness, outfile=output_file_path)
        for nneighbors in (3, 4):
            with pytest.raises(RuntimeError) as excinfo:
                pipeline.hxrecon()
            print(excinfo.value)
            pipeline.hxrecon(infile=file_path, suffix=f'recon_nn{nneighbors}')
