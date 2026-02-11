# Copyright (C) 2026 the hexsample team.
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

"""Support for legacy data.
"""

import struct

import numpy as np

import hexsample.xpol as xpol
from hexsample.digi import DigiEventRectangular
from hexsample.fileio import DigiOutputFileRectangular
from hexsample.logging_ import logger
from hexsample.roi import Padding, RegionOfInterest
from hexsample.readout import ReadoutProxy



class MDAT3File:

    """Class to read .mdat3 files.

    This is a legacy format whose basic structure is as follow

    - \xff\xff (event marker)
    - xmin, xmax, ymin, ymax (ROI, 16 bytes per number)
    - trigger ID (32 bytes)
    - livetime (32 bytes)
    - microseconds (32 bytes)
    - seconds (32 bytes)
    - status word (16 bytes)
    - error summary (16 bytes)
    - PHA values (16 bytes per pixel, in row-major order)
    """

    EVENT_MARKER = b"\xff\xff"
    DEFAULT_PADDING = Padding(7, 4, 4, 4)

    def __init__(self, file_path: str) -> None:
        """
        """
        self.file_path = file_path
        # We don't seem to keep track of the padding in the header, and I am
        # making up what I thing I understand from the data.
        self._file_handle = None
        self._padding = self.DEFAULT_PADDING
        self._eof = False
        self.header = None

    def __enter__(self) -> "MDAT3File":
        """Context manager entry point.
        """
        logger.info(f"Opening .mdat3 file '{self.file_path}'...")
        self._file_handle = open(self.file_path, "rb")
        self._eof = False
        self._header = self._read_header()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Context manager exit point.
        """
        if self._file_handle is not None:
            logger.info(f"Closing .mdat3 file '{self.file_path}'...")
            self._file_handle.close()
        self._file_handle = None
        self._eof = True

    def _read_header(self, delimiter: bytes = b";", encoding: str = "utf-8") -> None:
        """Read the header of the .mdat3 file.
        """
        header_bytes = bytearray()
        while True:
            byte = self._file_handle.read(1)
            if not byte:
                raise ValueError(f"'{delimiter.decode()}' not found in file")
            if byte == delimiter:
                break
            header_bytes.extend(byte)
        return header_bytes.decode(encoding)

    def _read16(self) -> int:
        """Read a 16-bit unsigned integer from the file.
        """
        return struct.unpack("H", self._file_handle.read(2))[0]

    def _read32(self) -> int:
        """Read a 32-bit unsigned integer from the file.
        """
        return struct.unpack("I", self._file_handle.read(4))[0]

    def _read_event(self) -> None:
        """Read an event from the .mdat3 file.
        """
        if self._file_handle.read(2) != self.EVENT_MARKER:
            raise ValueError("event header not found")
        xmin, xmax, ymin, ymax = self._read16(), self._read16(), self._read16(), self._read16()
        roi = RegionOfInterest(xmin, xmax, ymin, ymax, self._padding)
        trigger_id = self._read32()
        livetime, microseconds, seconds = self._read32(), self._read32(), self._read32()
        # We are not using the status word and error summary, but we need to read
        # them to move the file pointer to the right position for the PHA values.
        _, _ = self._read16(), self._read16()
        pha = np.fromfile(self._file_handle, dtype=np.uint16, count=roi.size)
        return DigiEventRectangular(trigger_id, seconds, microseconds, livetime, pha, roi)

    def __iter__(self) -> "MDAT3File":
        """Implementation of the iterator protocol.
        """
        return self

    def __next__(self):
        """Implementation of the iterator protocol.
        """
        if self._eof:
            raise StopIteration
        try:
            return self._read_event()
        except ValueError:
            self._eof = True
            raise StopIteration


def mdat3_to_digi(file_path: str, num_events: int = None) -> None:
    """Convert a .mdat3 file to a HDF5 digi file.

    This is a utility function that we can use to convert legacy data in the .mdat3
    format to the more modern HDF5 format.
    """
    if not file_path.endswith(".mdat3"):
        raise ValueError("input file must have .mdat3 extension")
    output_file_path = file_path.replace(".mdat3", ".h5")
    output_file = DigiOutputFileRectangular(output_file_path)
    # We are hardcoding the readout layout here
    readout_dict = dict(readout_mode="rectangular",
                        layout=xpol.XPOL3_LAYOUT,
                        num_cols=xpol.XPOL3_SIZE[0],
                        num_rows=xpol.XPOL3_SIZE[1],
                        pitch=xpol.XPOL_PITCH,
                        enc=None,
                        gain=None,
                        padding=MDAT3File.DEFAULT_PADDING)
    output_file.update_header(**readout_dict)
    with MDAT3File(file_path) as input_file:
        for i, event in enumerate(input_file):
            if i == num_events:
                #output_file.flush()
                output_file.close()
                break
            output_file.add_row(event, None)
    #output_file.flush()
    output_file.close()
