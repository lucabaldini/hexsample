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

"""Definition of the file format.
"""


from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Self
from unicodedata import name

from tables import group

import h5py
import numpy as np
import numpy.typing as npt


FILE_FORMAT = "pixel-detector"
CURRENT_SCHEMA_VERSION = "1.0.0"


# This is a placeholder, and we shall replace it with our own file structures.
@dataclass
class Event:

    trigger_id: int
    timestamp: int
    pha: npt.NDArray[np.uint16]



class SchemaError(RuntimeError):
    """The HDF5 file does not have a supported schema."""



class AbstractReadoutCodec(ABC):

    """Abstract base class for encoding and decoding detector data with
    a given readout mode (e.g., circular of rectangular).
    """

    _DEFAULT_CHUNK_SIZE = 4096

    mode: ClassVar[str]
    event_type: ClassVar[type]

    def _create_dataset(self, group: h5py.Group, name: str, dtype: np.dtype,
        shape: tuple[int, ...] = (0,), chunks: tuple[int, ...] = (_DEFAULT_CHUNK_SIZE,),
        maxshape: tuple[int, ...] = (None,), fill_data: Any = None, **kwargs) -> h5py.Dataset:
        """Create a dataset for a given group.

        Arguments
        ---------
        group: h5py.Group
            The HDF5 group in which to create the dataset.

        name: str
            The name of the dataset to create.

        dtype: np.dtype
            The data type of the dataset to create.

        shape: tuple[int, ...], optional
            The initial shape of the dataset. Defaults to (0,), which means an
            empty column that can grow later.

        chunks: tuple[int, ...], optional
            The chunk shape of the dataset.

        maxshape: tuple[int, ...], optional
            The maximum shape of the dataset. Defaults to (None,), which means
            that the dataset can be resized to any shape.

        fill_data: Any, optional
            The optional data to fill the dataset with.

        kwargs: dict
            Additional keyword arguments to pass to h5py.Group.create_dataset().
        """
        data_set = group.create_dataset(name, dtype=dtype, shape=shape, chunks=chunks, maxshape=maxshape, **kwargs)
        if fill_data is not None:
            data_set.resize((1,))
            data_set[0] = fill_data
        return data_set

    @abstractmethod
    def create_datasets(self, group: h5py.Group) -> None:
        """
        """

    @abstractmethod
    def append(self, group: h5py.Group, event: Any) -> None:
        """Append one event.
        """

    @abstractmethod
    def read(self, group: h5py.Group, index: int) -> Any:
        """Read one event.
        """

    @abstractmethod
    def event_count(self, group: h5py.Group) -> int:
        """Return the number of complete events.
        """





class RectangularCodec(AbstractReadoutCodec):

    mode = "rectangular"
    event_type = Event

    def create_datasets(self, group: h5py.Group) -> None:
        """
        """
        self._create_dataset(group, "trigger_id", np.uint64)
        self._create_dataset(group, "timestamp", np.uint64)
        self._create_dataset(group, "pha_offset", dtype=np.uint64, fill_data=0)
        self._create_dataset(group, "pha", dtype=np.uint16, compression="gzip", shuffle=True)

    def append(self, group: h5py.Group, event: Event) -> None:
        if not isinstance(event, Event):
            raise TypeError(
                f"{self.__class__.__name__} requires Event instances"
            )

        trigger_id = group["trigger_id"]
        timestamp = group["timestamp"]
        pha_offset = group["pha_offset"]
        pha = group["pha"]

        event_index = trigger_id.shape[0]
        pha_begin = int(pha_offset[-1])
        pha_end = pha_begin + event.pha.size

        # Write the variable-sized payload first. The scalar datasets and final
        # offset are extended afterwards, so their length acts as the committed
        # event count.
        pha.resize((pha_end,))
        pha[pha_begin:pha_end] = event.pha

        trigger_id.resize((event_index + 1,))
        timestamp.resize((event_index + 1,))
        pha_offset.resize((event_index + 2,))

        trigger_id[event_index] = event.trigger_id
        timestamp[event_index] = event.timestamp
        pha_offset[event_index + 1] = pha_end

    def read(self, group: h5py.Group, index: int) -> Event:
        count = self.event_count(group)

        if index < 0:
            index += count

        if not 0 <= index < count:
            raise IndexError(index)

        offsets = group["pha_offset"]
        begin = int(offsets[index])
        end = int(offsets[index + 1])

        return Event(
            trigger_id=int(group["trigger_id"][index]),
            timestamp=int(group["timestamp"][index]),
            pha=np.asarray(
                group["pha"][begin:end],
                dtype=np.uint16,
            ),
        )

    def event_count(self, group: h5py.Group) -> int:
        return int(group["trigger_id"].shape[0])


CODECS: dict[str, ReadoutCodec] = {
    RectangularCodec.mode: RectangularCodec(),
}


class PixelFile:
    """Reader/writer for versioned pixel-detector HDF5 files."""

    def __init__(
        self,
        path: str | Path,
        mode: str = "r",
        *,
        readout_mode: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.mode = mode
        self._requested_readout_mode = readout_mode
        self._metadata = dict(metadata or {})

        self._file: h5py.File | None = None
        self._group: h5py.Group | None = None
        self._codec: ReadoutCodec | None = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __len__(self) -> int:
        return self.codec.event_count(self.group)

    def __iter__(self) -> Iterator[Any]:
        # Iteration is lazy: one event payload is loaded at a time.
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> Any:
        return self.codec.read(self.group, index)

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise RuntimeError("file is not open")
        return self._file

    @property
    def group(self) -> h5py.Group:
        if self._group is None:
            raise RuntimeError("file is not open")
        return self._group

    @property
    def codec(self) -> ReadoutCodec:
        if self._codec is None:
            raise RuntimeError("file is not open")
        return self._codec

    @property
    def readout_mode(self) -> str:
        return self.codec.mode

    @property
    def schema_version(self) -> str:
        return str(self.file.attrs["schema_version"])

    def open(self) -> None:
        if self._file is not None:
            raise RuntimeError("file is already open")

        creating = self.mode in {"w", "w-", "x"}

        try:
            self._file = h5py.File(self.path, self.mode)

            if creating:
                self._initialize_new_file()
            else:
                self._open_existing_file()

        except Exception:
            if self._file is not None:
                self._file.close()
            self._file = None
            self._group = None
            self._codec = None
            raise

    def close(self) -> None:
        if self._file is not None:
            self._file.close()

        self._file = None
        self._group = None
        self._codec = None

    def flush(self) -> None:
        self.file.flush()

    def append(self, event: Any) -> None:
        if self.mode == "r":
            raise OSError("file was opened read-only")

        self.codec.append(self.group, event)

    def _initialize_new_file(self) -> None:
        if self._requested_readout_mode is None:
            raise ValueError(
                "readout_mode is required when creating a file"
            )

        try:
            codec = CODECS[self._requested_readout_mode]
        except KeyError as exc:
            raise ValueError(
                f"unsupported readout mode: "
                f"{self._requested_readout_mode!r}"
            ) from exc

        self.file.attrs["file_format"] = FILE_FORMAT
        self.file.attrs["schema_version"] = CURRENT_SCHEMA_VERSION
        self.file.attrs["readout_mode"] = codec.mode

        for name, value in self._metadata.items():
            if name in {
                "file_format",
                "schema_version",
                "readout_mode",
            }:
                raise ValueError(f"reserved metadata name: {name!r}")
            self.file.attrs[name] = value

        self._group = self.file.create_group("events")
        codec.create_datasets(self.group)
        self._codec = codec

    def _open_existing_file(self) -> None:
        file_format = self.file.attrs.get("file_format")
        schema_version = self.file.attrs.get("schema_version")
        readout_mode = self.file.attrs.get("readout_mode")

        if file_format != FILE_FORMAT:
            raise SchemaError(
                f"not a {FILE_FORMAT!r} file: {file_format!r}"
            )

        if schema_version != CURRENT_SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema version: {schema_version!r}"
            )

        if not isinstance(readout_mode, str):
            raise SchemaError("missing or invalid readout_mode")

        try:
            self._codec = CODECS[readout_mode]
        except KeyError as exc:
            raise SchemaError(
                f"unsupported readout mode: {readout_mode!r}"
            ) from exc

        try:
            self._group = self.file["events"]
        except KeyError as exc:
            raise SchemaError("missing /events group") from exc




# Define a list of events.

events = [
    Event(
        trigger_id=100,
        timestamp=1_720_000_000_000_000_000,
        pha=np.array([12, 18, 9], dtype=np.uint16),
    ),
    Event(
        trigger_id=101,
        timestamp=1_720_000_000_000_001_000,
        pha=np.array([7, 21, 15, 2, 4], dtype=np.uint16),
    ),
]


# Write a file.
with PixelFile(
    "run001.h5",
    "w",
    readout_mode="rectangular",
    metadata={
        "detector_name": "prototype-01",
        "acquisition_software": "pixdaq 0.1.0",
    },
) as output:
    for event in events:
        output.append(event)

# Read back the file.
with PixelFile("run001.h5") as input_file:

    ts = input_file.group["trigger_id"]
    print(type(ts), ts, ts[:])

    for event in input_file:
        print(event)
