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

"""Definition of the file format.
"""

from enum import Enum
import inspect
import pathlib
import time
from typing import Any

from loguru import logger
import numpy as np
import tables

from hexsample import __version__, __tagdate__
from hexsample.mc import MonteCarloEvent
from hexsample.digi import DigiEvent
from hexsample.recon import ReconEvent


class MonteCarloDescription(tables.IsDescription):

    """Description of the Monte Carlo part of the file format.
    """

    # pylint: disable=too-few-public-methods

    timestamp = tables.Float64Col(pos=0)
    energy = tables.Float32Col(pos=1)
    absx = tables.Float32Col(pos=2)
    absy = tables.Float32Col(pos=3)
    absz = tables.Float32Col(pos=4)
    num_pairs = tables.Int32Col(pos=5)

def _fill_mc_row(row : tables.tableextension.Row, event : MonteCarloEvent) -> None:
    """Helper function to fill an output table row, given a MonteCarloEvent object.

    .. note::
       This would have naturally belonged to the MonteCarloDescription class as
       a @staticmethod, but doing so is apparently breaking something into the
       tables internals, and all of a sudden you get an exception due to the
       fact that a staticmethod cannot be pickled.
    """
    row['timestamp'] = event.timestamp
    row['energy'] = event.energy
    row['absx'] = event.absx
    row['absy'] = event.absy
    row['absz'] = event.absz
    row['num_pairs'] = event.num_pairs
    row.append()



class DigiDescription(tables.IsDescription):

    """Description of the (flat) digi part of the file format.
    """

    # pylint: disable=too-few-public-methods

    trigger_id = tables.Int32Col(pos=0)
    seconds = tables.Int32Col(pos=1)
    microseconds = tables.Int32Col(pos=2)
    livetime = tables.Int32Col(pos=3)
    min_col = tables.Int16Col(pos=4)
    max_col = tables.Int16Col(pos=5)
    min_row = tables.Int16Col(pos=6)
    max_row = tables.Int16Col(pos=7)
    padding_top = tables.Int8Col(pos=8)
    padding_right = tables.Int8Col(pos=9)
    padding_bottom = tables.Int8Col(pos=10)
    padding_left = tables.Int8Col(pos=11)

def _fill_digi_row(row : tables.tableextension.Row, event : DigiEvent) -> None:
    """Helper function to fill an output table row, given a DigiEvent object.

    .. note::
       This would have naturally belonged to the DigiDescription class as
       a @staticmethod, but doing so is apparently breaking something into the
       tables internals, and all of a sudden you get an exception due to the
       fact that a staticmethod cannot be pickled.
    """
    row['trigger_id'] = event.trigger_id
    row['seconds'] = event.seconds
    row['microseconds'] = event.microseconds
    row['livetime'] = event.livetime
    row['min_col'] = event.roi.min_col
    row['max_col'] = event.roi.max_col
    row['min_row'] = event.roi.min_row
    row['max_row'] = event.roi.max_row
    row['padding_top'] = event.roi.padding.top
    row['padding_right'] = event.roi.padding.right
    row['padding_bottom'] = event.roi.padding.bottom
    row['padding_left'] = event.roi.padding.left
    row.append()



class ReconDescription(tables.IsDescription):

    """Description of the recon file format.
    """

    # pylint: disable=too-few-public-methods

    trigger_id = tables.Int32Col(pos=0)
    timestamp = tables.Float64Col(pos=1)
    livetime = tables.Int32Col(pos=2)
    roi_size = tables.Int32Col(pos=3)
    cluster_size = tables.Int8Col(pos=4)
    energy = tables.Float32Col(pos=5)
    posx = tables.Float32Col(pos=6)
    posy = tables.Float32Col(pos=7)

def _fill_recon_row(row : tables.tableextension.Row, event : ReconEvent) -> None:
    """Helper function to fill an output table row, given a ReconEvent object.

    .. note::
       This would have naturally belonged to the ReconDescription class as
       a @staticmethod, but doing so is apparently breaking something into the
       tables internals, and all of a sudden you get an exception due to the
       fact that a staticmethod cannot be pickled.
    """
    row['trigger_id'] = event.trigger_id
    row['timestamp'] = event.timestamp
    row['livetime'] = event.livetime
    row['roi_size'] = event.roi_size
    row['cluster_size'] = event.cluster.size()
    row['energy'] = event.energy()
    row['posx'], row['posy'] = event.position()
    row.append()



class FileType(Enum):

    """Enum class for the different file types.
    """

    DIGI = 'Digi'
    RECON = 'Recon'



class OutputFileBase(tables.File):

    """Base class for output files.

    The base class has the responsibility of opening the output file and create a
    header node to store all the necessary metadata. Subclasses can use the
    update_header() hook to write arbitrary user attributes in the header node.

    Note this is a purely virtual class, and subclasses should reimplement the
    add_row() and flush() methods.

    Arguments
    ---------
    file_path : str
        The path to the file on disk.
    """

    _DATE_FORMAT = '%a, %d %b %Y %H:%M:%S %z'
    _FILE_TYPE = None

    def __init__(self, file_path : str) -> None:
        """Constructor.
        """
        logger.info(f'Opening output file {file_path}...')
        super().__init__(file_path, 'w')
        self.header_group = self.create_group(self.root, 'header', 'File header')
        date = time.strftime(self._DATE_FORMAT)
        creator = pathlib.Path(inspect.stack()[-1].filename).name
        self.update_header(filetype=self._FILE_TYPE.value, date=date, creator=creator,
            version=__version__, tagdate=__tagdate__)

    def update_header(self, **kwargs) -> None:
        """Update the user attributes in the header group.
        """
        self.update_user_attributes(self.header_group, **kwargs)

    @staticmethod
    def _set_user_attribute(group : tables.group.Group, name : str, value : Any) -> None:
        """Set a user attribute for a given group.
        """
        # pylint: disable=protected-access
        group._v_attrs[name] = value

    @staticmethod
    def update_user_attributes(group : tables.group.Group, **kwargs) -> None:
        """Update the user attributes for a given group.

        The basic rules, here, are that all the keys of the keyword arguments
        must be string, and the values can be arbitrary data types. Following
        up on the discussion at https://www.pytables.org/usersguide/tutorials.html
        we write the keywords arguments one at a time (as opposed to the entire
        dictionary all at once) and make an effort to convert the Python types
        to native numpy arrays when that is not performed automatically (e.g.,
        for lists and tuples). This avoids the need for serializing the Python
        data and should ensure that the output file can be read with any (be it
        Python-aware or not) application.
        """
        # pylint: disable=protected-access
        logger.info(f'Updating {group._v_pathname} group user attributes...')
        for name, value in kwargs.items():
            if isinstance(value, (tuple, list)):
                logger.debug(f'Converting {name} ({value}) to a native numpy array...')
                value = np.array(value)
                logger.debug(f'-> {value}.')
            if value is None:
                logger.debug(f'Converting {name} ({value}) to string...')
                value = str(value)
                logger.debug(f'-> {value}.')
            OutputFileBase._set_user_attribute(group, name, value)

    def add_row(self, *args) -> None:
        """Virtual function to add a row to the output file.

        This needs to be reimplemented in derived classes.
        """
        raise NotImplementedError

    def flush(self) -> None:
        """Virtual function to flush the file.

        This needs to be reimplemented in derived classes.
        """
        raise NotImplementedError



class DigiOutputFile(OutputFileBase):

    """Description of a digitized output file.

    This can represent either a digitized files written by the DAQ, or the
    output of a simulation, in which case it contains an additional group and
    table for the Monte Carlo ground truth.

    Arguments
    ---------
    file_path : str
        The path to the file on disk.
    """

    _FILE_TYPE = FileType.DIGI
    DIGI_TABLE_SPECS = ('digi_table', DigiDescription, 'Digi data')
    PHA_ARRAY_SPECS = ('pha', tables.Int32Atom(shape=()))
    MC_TABLE_SPECS = ('mc_table', MonteCarloDescription, 'Monte Carlo data')

    def __init__(self, file_path : str):
        """Constructor.
        """
        super().__init__(file_path)
        self.digi_group = self.create_group(self.root, 'digi', 'Digi')
        self.digi_table = self.create_table(self.digi_group, *self.DIGI_TABLE_SPECS)
        self.pha_array = self.create_vlarray(self.digi_group, *self.PHA_ARRAY_SPECS)
        self.mc_group = self.create_group(self.root, 'mc', 'Monte Carlo')
        self.mc_table = self.create_table(self.mc_group, *self.MC_TABLE_SPECS)

    def add_row(self, digi_event : DigiEvent, mc_event : MonteCarloEvent) -> None:
        """Add one row to the file.

        Arguments
        ---------
        digi : DigiEvent
            The digitized event contribution.

        mc : MonteCarloEvent
            The Monte Carlo event contribution.
        """
        # pylint: disable=arguments-differ
        _fill_digi_row(self.digi_table.row, digi_event)
        self.pha_array.append(digi_event.pha.flatten())
        _fill_mc_row(self.mc_table.row, mc_event)

    def flush(self) -> None:
        """Flush the basic file components.
        """
        self.digi_table.flush()
        self.pha_array.flush()
        self.mc_table.flush()



class ReconOutputFile(OutputFileBase):

    """Description of a reconstructed output file.

    Arguments
    ---------
    file_path : str
        The path to the file on disk.
    """

    _FILE_TYPE = FileType.RECON
    RECON_TABLE_SPECS = ('recon_table', ReconDescription, 'Recon data')
    MC_TABLE_SPECS = ('mc_table', MonteCarloDescription, 'Monte Carlo data')

    def __init__(self, file_path : str):
        """Constructor.
        """
        super().__init__(file_path)
        self.digi_header_group = self.create_group(self.root, 'digi_header', 'Digi file header')
        self.recon_group = self.create_group(self.root, 'recon', 'Recon')
        self.recon_table = self.create_table(self.recon_group, *self.RECON_TABLE_SPECS)
        self.mc_group = self.create_group(self.root, 'mc', 'Monte Carlo')
        self.mc_table = self.create_table(self.mc_group, *self.MC_TABLE_SPECS)

    def update_digi_header(self, **kwargs):
        """Update the user arguments in the digi header group.
        """
        self.update_user_attributes(self.digi_header_group, **kwargs)

    def add_row(self, recon_event : ReconEvent, mc_event : MonteCarloEvent) -> None:
        """Add one row to the file.

        Arguments
        ---------
        digi : DigiEvent
            The digitized event contribution.

        mc : MonteCarloEvent
            The Monte Carlo event contribution.
        """
        # pylint: disable=arguments-differ
        _fill_recon_row(self.recon_table.row, recon_event)
        _fill_mc_row(self.mc_table.row, mc_event)

    def flush(self) -> None:
        """Flush the basic file components.
        """
        self.recon_table.flush()
        self.mc_table.flush()



class InputFileBase(tables.File):

    """Base class for input files.
    """

    def __init__(self, file_path : str):
        """Constructor.
        """
        logger.info(f'Opening input file {file_path}...')
        super().__init__(file_path, 'r')
        self.header = self._user_attributes(self.root.header)
        self.file_type = FileType(self.header_value('filetype'))
        logger.info(f'File type: {self.file_type}')

    @staticmethod
    def _user_attributes(group : tables.group.Group) -> dict:
        """Return all the user attributes for a given group in the form of a
        Python dictionary.

        This is used, e.g, to rebuild the header information.
        """
        # pylint: disable=protected-access
        return {key : group._v_attrs[key] for key in group._v_attrs._f_list('user')}

    def header_value(self, key : str, default : Any = None) -> Any:
        """Return the header value corresponding to a given key.
        """
        return self.header.get(key, default)



class DigiInputFile(InputFileBase):

    """Description of a digitized input file.

    This has a very simple interface: we cache references to the relevant tables
    when we open the file and we provide methods to reassemble a specific table
    row into the corresponding DigiEvent or MonteCarloEvent objects, along with
    an implementation of the iterator protocol to make event loops easier.
    """

    def __init__(self, file_path : str):
        """Constructor.
        """
        super().__init__(file_path)
        self.digi_table = self.root.digi.digi_table
        self.pha_array = self.root.digi.pha
        self.mc_table = self.root.mc.mc_table
        self.__index = -1

    def digi_event(self, row_index : int) -> DigiEvent:
        """Random access to the DigiEvent part of the event contribution.

        Arguments
        ---------
        row_index : int
            The index of the target row in the event file.
        """
        row = self.digi_table[row_index]
        pha = self.pha_array[row_index]
        return DigiEvent.from_digi(row, pha)

    def mc_event(self, row_index : int) -> MonteCarloEvent:
        """Random access to the MonteCarloEvent part of the event contribution.

        Arguments
        ---------
        row_index : int
            The index of the target row in the event file.
        """
        row =  self.mc_table[row_index]
        return MonteCarloEvent(*row)

    def __iter__(self):
        """Overloaded method for the implementation of the iterator protocol.
        """
        self.__index = -1
        return self

    def __next__(self) -> DigiEvent:
        """Overloaded method for the implementation of the iterator protocol.
        """
        self.__index += 1
        if self.__index == len(self.digi_table):
            raise StopIteration
        return self.digi_event(self.__index)



class ReconInputFile(InputFileBase):

    """Description of a reconstructed input file.
    """

    def __init__(self, file_path : str):
        """Constructor.
        """
        super().__init__(file_path)
        self.digi_header = self._user_attributes(self.root.digi_header)
        self.recon_table = self.root.recon.recon_table
        self.mc_table = self.root.mc.mc_table

    def recon_column(self, name : str) -> np.ndarray:
        """Return a given column in the recon table.
        """
        return self.recon_table.col(name)

    def mc_column(self, name : str) -> np.ndarray:
        """Return a given column in the Monte Carlo table.
        """
        return self.mc_table.col(name)



def peek_file_type(file_path : str) -> FileType:
    """Peek into the header of a HDF5 file and determing the file type.

    Arguments
    ---------
    file_path : str
        The path to the input file.
    """
    with tables.open_file(file_path, 'r') as input_file:
        try:
            return FileType(input_file.root.header._v_attrs['filetype'])
        except KeyError:
            raise RuntimeError(f'File {file_path} has no type information.')

def open_input_file(file_path : str) -> InputFileBase:
    """Open an input file automatically determining the file type.

    Arguments
    ---------
    file_path : str
        The path to the output file.
    """
    file_type = peek_file_type(file_path)
    if file_type == FileType.DIGI:
        return DigiInputFile(file_path)
    if file_type == FileType.RECON:
        return ReconInputFile(file_path)
    raise RuntimeError(f'Invalid input file type {file_type}')
>>>>>>> main
