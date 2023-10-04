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

from loguru import logger
import tables

from hexsample.mc import MonteCarloEvent
from hexsample.digi import DigiEvent
from hexsample.roi import Padding, RegionOfInterest


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



class DigiOutputFile(tables.File):

    """Description of a digitized output file.

    This can represent either a digitized files written by the DAQ, or the
    output of a simulation, in which case it contains an additional group and
    table for the Monte Carlo ground truth.

    Arguments
    ---------
    file_path : str
        The path to the file on disk.

    mc : bool
        If True, a specific group and table is created in the file to hold the
        Monte Carlo information.
    """

    DIGI_TABLE_SPECS = ('digi_table', DigiDescription, 'Digi data')
    PHA_ARRAY_SPECS = ('pha', tables.Int32Atom(shape=()))
    MC_TABLE_SPECS = ('mc_table', MonteCarloDescription, 'Monte Carlo data')

    def __init__(self, file_path : str, mc : bool = False):
        """Constructor.
        """
        logger.info(f'Opening output digi file {file_path}...')
        super().__init__(file_path, 'w')
        #self.config = self.create_group(self.root, 'config', 'Data acquisition setup')
        self.digi_group = self.create_group(self.root, 'digi', 'Digi')
        self.digi_table = self.create_table(self.digi_group, *self.DIGI_TABLE_SPECS)
        self.pha_array = self.create_vlarray(self.digi_group, *self.PHA_ARRAY_SPECS)
        if mc:
            self.mc_group = self.create_group(self.root, 'mc', 'Monte Carlo')
            self.mc_table = self.create_table(self.mc_group, *self.MC_TABLE_SPECS)
        else:
            self.mc_group = None
            self.mc_table = None

    def add_row(self, digi_event : DigiEvent, mc_event : MonteCarloEvent = None) -> None:
        """Add one row to the file.

        Arguments
        ---------
        digi : DigiEvent
            The digitized event contribution.

        mc : MonteCarloEvent
            The Monte Carlo event contribution.
        """
        row = self.digi_table.row
        row['trigger_id'] = digi_event.trigger_id
        row['seconds'] = digi_event.seconds
        row['microseconds'] = digi_event.microseconds
        row['min_col'] = digi_event.roi.min_col
        row['max_col'] = digi_event.roi.max_col
        row['min_row'] = digi_event.roi.min_row
        row['max_row'] = digi_event.roi.max_row
        row['padding_top'] = digi_event.roi.padding.top
        row['padding_right'] = digi_event.roi.padding.right
        row['padding_bottom'] = digi_event.roi.padding.bottom
        row['padding_left'] = digi_event.roi.padding.left
        row.append()
        self.pha_array.append(digi_event.pha.flatten())
        if mc_event is not None:
            row = self.mc_table.row
            row['timestamp'] = mc_event.timestamp
            row['energy'] = mc_event.energy
            row['absx'] = mc_event.absx
            row['absy'] = mc_event.absy
            row['absz'] = mc_event.absz
            row['num_pairs'] = mc_event.num_pairs
            row.append()

    def flush(self) -> None:
        """Flush the basic file components.
        """
        self.digi_table.flush()
        self.pha_array.flush()
        if self.mc_table is not None:
            self.mc_table.flush()



class DigiInputFile(tables.File):

    """Description of a digitized input file.

    This has a very simple interface: we cache references to the relevant tables
    when we open the file and we provide methods to reassemble a specific table
    row into the corresponding DigiEvent or MonteCarloEvent objects, along with
    an implementation of the iterator protocol to make event loops easier.
    """

    def __init__(self, file_path : str):
        """Constructor.
        """
        logger.info(f'Opening input digi file {file_path}...')
        super().__init__(file_path, 'r')
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
