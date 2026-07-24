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

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import h5py
import numpy as np


@dataclass
class DatasetSpec:

    """Dataset specification.

    See https://docs.h5py.org/en/stable/high/dataset.html for more information.

    Arguments
    ---------
    path: str
        The path of the dataset within the HDF5 file. (This is called ``name`` in h5py.

    dtype: Any
        The data type of the dataset. This can be any valid NumPy dtype or a string
        that can be converted to a NumPy dtype.

    shape: Tuple
        The initial shape of the dataset. Default is (0,).

    maxshape: Tuple
        The maximum shape of the dataset. Default is (None,), which means that the
        dataset can be resized to any shape.

    chunks: Optional[Tuple[int, ...]]
        The chunk shape of the dataset. Default is (8192,). If None, the dataset
        will not be chunked.

    compression: Optional[str]
        The compression algorithm to use for the dataset. Default is "lzf".
        If None, the dataset will not be compressed.

    shuffle: bool
        Whether to apply the shuffle filter to the dataset. Default is True.

    attrs: dict[str, Any]
        A dictionary of attributes to set on the dataset.
    """

    path: str
    dtype: Any
    shape: Tuple = (0,)
    maxshape: Tuple = (None,)
    chunks: Optional[Tuple[int, ...]] = (8192,)
    compression: Optional[str] = "lzf"
    shuffle: bool = True
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupSpec:

    """Group specification.

    See https://docs.h5py.org/en/stable/high/group.html for more information.

    Arguments
    ---------
    path: str
        The path of the group within the HDF5 file. (This is called ``name`` in h5py.)

    attrs: dict[str, Any]
        A dictionary of attributes to set on the group.
    """

    path: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class FileSchema:

    """File schema specification.

    Arguments
    ---------
    name: str
        The name of the file format.

    variant: str
        The variant of the file format.

    version: int
        The version of the file format.

    groups: list[GroupSpec]
        A list of group specifications.

    datasets: list[DatasetSpec]
        A list of dataset specifications.

    attrs: dict[str, Any]
        A dictionary of attributes to set on the root group of the HDF5 file.
    """

    name: str
    variant: str
    version: int
    groups: list[GroupSpec]
    datasets: list[DatasetSpec]
    attrs: dict[str, Any] = field(default_factory=dict)


class HDF5Backend:

    """HDF5 backend for writing data according to a specified file schema.

    The backend is format-agnostic and knows nothing about the semantics of the
    data. It simply open/close files, creates the groups and the datasets according
    to the provided schema and provides a method for appending data to the datasets.
    """

    def __init__(self, file_path, schema):
        self.file_path = file_path
        self.schema = schema
        self.datasets = {}

    def open(self):
        self.f = h5py.File(self.file_path, "w")

        for k, v in self.schema.attrs.items():
            self.f.attrs[k] = v

        for group in self.schema.groups:
            g = self.f.require_group(group.path)
            for k, v in group.attrs.items():
                g.attrs[k] = v

        for ds in self.schema.datasets:
            dset = self.f.create_dataset(
                ds.path,
                shape=ds.shape,
                maxshape=ds.maxshape,
                dtype=ds.dtype,
                chunks=ds.chunks,
                compression=ds.compression,
                shuffle=ds.shuffle,
            )
            for k, v in ds.attrs.items():
                dset.attrs[k] = v
            self.datasets[ds.path] = dset

    def append_1d(self, path, values):
        ds = self.datasets[path]
        n = ds.shape[0]
        m = len(values)
        ds.resize((n + m,))
        ds[n:n + m] = values

    def close(self):
        self.f.close()


class EventWriter:
    def __init__(self, path):
        self.f = h5py.File(path, "w")

        # File metadata
        self.f.attrs["format_name"] = "pixel-detector-events"
        self.f.attrs["format_version"] = 1
        self.f.attrs["timestamp_unit"] = "ns since unix epoch"
        self.f.attrs["pha_unit"] = "ADC counts"

        events = self.f.create_group("events")
        hits = self.f.create_group("hits")

        self.ds_timestamp = events.create_dataset(
            "timestamp",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint64,
            chunks=(8192,),
            compression="gzip",
            shuffle=True,
        )
        self.ds_trigger_id = events.create_dataset(
            "trigger_id",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint32,
            chunks=(8192,),
            compression="gzip",
            shuffle=True,
        )
        self.ds_event_id = events.create_dataset(
            "event_id",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint64,
            chunks=(8192,),
            compression="gzip",
            shuffle=True,
        )
        self.ds_pha_offset = events.create_dataset(
            "pha_offset",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint64,
            chunks=(8192,),
            compression="gzip",
            shuffle=True,
        )
        self.ds_pha_count = events.create_dataset(
            "pha_count",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint32,
            chunks=(8192,),
            compression="gzip",
            shuffle=True,
        )

        self.ds_pha = hits.create_dataset(
            "pha",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint16,
            chunks=(65536,),
            compression="gzip",
            shuffle=True,
        )
        self.ds_pixel_id = hits.create_dataset(
            "pixel_id",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint16,
            chunks=(65536,),
            compression="gzip",
            shuffle=True,
        )

    def append_event(self, timestamp, trigger_id, event_id, pha, pixel_id=None):
        pha = np.asarray(pha, dtype=np.uint16)

        if pixel_id is None:
            pixel_id = np.zeros(len(pha), dtype=np.uint16)
        else:
            pixel_id = np.asarray(pixel_id, dtype=np.uint16)

        if len(pixel_id) != len(pha):
            raise ValueError("pixel_id must have the same length as pha")

        n_events = self.ds_timestamp.shape[0]
        n_hits = self.ds_pha.shape[0]
        n_new_hits = len(pha)

        # Extend event datasets
        new_n_events = n_events + 1
        self.ds_timestamp.resize((new_n_events,))
        self.ds_trigger_id.resize((new_n_events,))
        self.ds_event_id.resize((new_n_events,))
        self.ds_pha_offset.resize((new_n_events,))
        self.ds_pha_count.resize((new_n_events,))

        self.ds_timestamp[n_events] = timestamp
        self.ds_trigger_id[n_events] = trigger_id
        self.ds_event_id[n_events] = event_id
        self.ds_pha_offset[n_events] = n_hits
        self.ds_pha_count[n_events] = n_new_hits

        # Extend hit datasets
        new_n_hits = n_hits + n_new_hits
        self.ds_pha.resize((new_n_hits,))
        self.ds_pixel_id.resize((new_n_hits,))

        self.ds_pha[n_hits:new_n_hits] = pha
        self.ds_pixel_id[n_hits:new_n_hits] = pixel_id

    def close(self):
        self.f.close()


class EventReader:
    def __init__(self, path):
        self.f = h5py.File(path, "r")
        self.events = self.f["events"]
        self.hits = self.f["hits"]

    def __len__(self):
        return len(self.events["timestamp"])

    def read_event(self, i):
        start = int(self.events["pha_offset"][i])
        count = int(self.events["pha_count"][i])
        stop = start + count

        return {
            "timestamp": int(self.events["timestamp"][i]),
            "trigger_id": int(self.events["trigger_id"][i]),
            "event_id": int(self.events["event_id"][i]),
            "pha": self.hits["pha"][start:stop],
            "pixel_id": self.hits["pixel_id"][start:stop],
        }

    def close(self):
        self.f.close()



if __name__ == "__main__":
    writer = EventWriter("events.h5")

    writer.append_event(
        timestamp=1710000000000000000,
        trigger_id=42,
        event_id=0,
        pha=[120, 98, 211],
        pixel_id=[5, 8, 12],
    )

    writer.append_event(
        timestamp=1710000000000001000,
        trigger_id=43,
        event_id=1,
        pha=[77, 88],
        pixel_id=[1, 3],
    )

    writer.close()