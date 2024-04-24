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

"""Test suite for hexsample.fileio
"""

from loguru import logger
import numpy as np

from hexsample import HEXSAMPLE_DATA
from hexsample.digi import DigiEventSparse, DigiEventRectangular, DigiEventCircular
from hexsample.readout import HexagonalReadoutMode, HexagonalReadoutCircular
from hexsample.fileio import DigiInputFile, DigiInputFileSparse, DigiInputFileRectangular,\
    DigiInputFileCircular, ReconInputFile, ReconOutputFile,FileType,\
    DigiOutputFile, DigiOutputFileSparse, DigiOutputFileRectangular, DigiOutputFileCircular,\
    peek_file_type, peek_readout_type, open_input_file, digioutput_class
from hexsample.mc import MonteCarloEvent
from hexsample.roi import RegionOfInterest, Padding



def _mc_event(index : int) -> MonteCarloEvent:
    """Create a bogus MonteCarloEvent object with index-dependent properties.
    """
    return MonteCarloEvent(0.1 * index, 5.9, 0., 0., 0.02, 1000 + index)

def _digi_event_rectangular(index : int) -> DigiEventRectangular:
    """Create a bogus DigiEvent object with index-dependent properties.
    """
    roi = RegionOfInterest(100, 107, 150, 155 + index * 2, Padding(2))
    pha = np.full(roi.size, index)
    return DigiEventRectangular(index, index, index, 0, pha, roi)

def _digi_event_sparse(index : int) -> DigiEventSparse:
    columns = np.array([0, 3, 6, 7], dtype=int)
    rows = np.array([0, 1, 2, 4], dtype=int)
    assert len(columns) == len(rows)
    pha = np.full(len(columns), index)
    return DigiEventSparse(index, index, index, 0, pha, columns, rows)

def _digi_event_circular(index : int) -> DigiEventCircular:
    column = index
    row = index
    pha = np.array([100, 200, 0, 75, 43, 31, 98])
    assert len(pha) == HexagonalReadoutCircular.NUM_PIXELS
    return DigiEventCircular(index, index, index, 0, pha, column, row)

def _test_write(file_path, num_events : int = 10):
    """Small test writing a bunch of toy event structures to file.
    This test considers a rectangular type of readout.
    """
    output_file = digioutput_class(HexagonalReadoutMode.RECTANGULAR)(file_path)
    for i in range(num_events):
        output_file.add_row(_digi_event_rectangular(i), _mc_event(i))
    output_file.close()

def _test_write_sparse(file_path, num_events : int = 10):
    """Small test writing a bunch of toy event structures to file.
    This test considers a sparse type of readout.
    """
    output_file = digioutput_class(HexagonalReadoutMode.SPARSE)(file_path)
    for i in range(num_events):
        output_file.add_row(_digi_event_sparse(i), _mc_event(i))
    output_file.close()

def _test_write_circular(file_path, num_events : int = 10):
    """Small test writing a bunch of toy event structures to file.
    This test considers a sparse type of readout.
    """
    output_file = digioutput_class(HexagonalReadoutMode.CIRCULAR)(file_path)
    for i in range(num_events):
        output_file.add_row(_digi_event_circular(i), _mc_event(i))
    output_file.close()

def _test_read(file_path):
    """Small test iterating over an input file.
    """
    input_file = DigiInputFile(file_path)
    print(input_file)
    for i, event in enumerate(input_file):
        print(event.ascii())
        print(input_file.mc_event(i))
        target = _digi_event_rectangular(i)
        assert event.trigger_id == target.trigger_id
        assert event.seconds == target.seconds
        assert event.microseconds == target.microseconds
        assert event.roi == target.roi
        assert (event.pha == target.pha).all()
    input_file.close()

def _test_read_sparse(file_path):
    """Small test iterating over a sparse input file.
    """
    input_file = DigiInputFileSparse(file_path)
    print(input_file)
    for i, event in enumerate(input_file):
        print(event.ascii())
        print(input_file.mc_event(i))
        target = _digi_event_sparse(i)
        assert event.trigger_id == target.trigger_id
        assert event.seconds == target.seconds
        assert event.microseconds == target.microseconds
        assert (event.columns == target.columns).all()
        assert (event.rows == target.rows).all()
        assert (event.pha == target.pha).all()
    input_file.close()

def _test_read_circular(file_path):
    """Small test iterating over a sparse input file.
    """
    input_file = DigiInputFileCircular(file_path)
    print(input_file)
    for i, event in enumerate(input_file):
        print(event.ascii())
        print(input_file.mc_event(i))
        target = _digi_event_circular(i)
        assert event.trigger_id == target.trigger_id
        assert event.seconds == target.seconds
        assert event.microseconds == target.microseconds
        assert event.column == target.column
        assert event.row == target.row
        assert (event.pha == target.pha).all()
    input_file.close()

def test():
    """Write and read back a simple digi file.
    """
    file_path = HEXSAMPLE_DATA / 'test_io_rectangular.h5'
    logger.info(f'Testing output file {file_path}...')
    _test_write(file_path)
    logger.info(f'Testing input file {file_path}...')
    _test_read(file_path)

def test_sparse():
    """Write and read back a sparse digi file.
    """
    file_path = HEXSAMPLE_DATA / 'test_io_sparse.h5'
    logger.info(f'Testing output file {file_path}...')
    _test_write_sparse(file_path)
    logger.info(f'Testing input file {file_path}...')
    _test_read_sparse(file_path)

def test_circular():
    """Write and read back a sparse digi file.
    """
    file_path = HEXSAMPLE_DATA / 'test_io_circular.h5'
    logger.info(f'Testing output file {file_path}...')
    _test_write_circular(file_path)
    logger.info(f'Testing input file {file_path}...')
    _test_read_circular(file_path)

def test_file_type():
    """Test the auto-recognition machinery for input file types.
    """
    # Test for the digi files.
    file_path = HEXSAMPLE_DATA / 'test_digi_filetype_rectangular.h5'
    digi_file = DigiOutputFileRectangular(file_path)
    digi_file.close()
    digi_file = DigiInputFileRectangular(file_path)
    assert digi_file.file_type == FileType.DIGI
    digi_file.close()
    assert peek_file_type(file_path) == FileType.DIGI
    digi_file = open_input_file(file_path)
    assert digi_file.file_type == FileType.DIGI
    digi_file.close()
    # Test for the digi sparse files.
    file_path = HEXSAMPLE_DATA / 'test_digi_filetype_sparse_readouttype.h5'
    digi_file = DigiOutputFileSparse(file_path)
    digi_file.close()
    digi_file = DigiInputFileSparse(file_path)
    assert digi_file.file_type == FileType.DIGI
    digi_file.close()
    assert peek_file_type(file_path) == FileType.DIGI
    assert peek_readout_type(file_path) == HexagonalReadoutMode.SPARSE
    digi_file = open_input_file(file_path)
    assert isinstance(digi_file, DigiInputFileSparse)
    assert digi_file.file_type == FileType.DIGI
    digi_file.close()
    # Test for the recon files.
    file_path = HEXSAMPLE_DATA / 'test_recon_filetype.h5'
    recon_file = ReconOutputFile(file_path)
    recon_file.close()
    recon_file = ReconInputFile(file_path)
    assert recon_file.file_type == FileType.RECON
    recon_file.close()
    assert peek_file_type(file_path) == FileType.RECON
    recon_file = open_input_file(file_path)
    assert isinstance(recon_file, ReconInputFile)
    assert recon_file.file_type == FileType.RECON
    recon_file.close()
