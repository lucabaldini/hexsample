# Copyright (C) 2023--2025 the hexsample team.
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

"""Pipeline facilities.
"""

from . import tasks
from .readout import ReadoutProxy
from .source import Source
from .sensor import Sensor


def simulate(**kwargs) -> str:
    """Run a simulation.

    .. warning::

       We still need to do something to make sure that we get a consistent behavior for
       number of events and output file path across the CLI and programmatic usage.
    """
    source = Source.from_filtered_kwargs(**kwargs)
    sensor = Sensor.from_filtered_kwargs(**kwargs)
    readout = ReadoutProxy.from_filtered_kwargs(**kwargs)
    num_events = kwargs.get("num_events", tasks.SimulationDefaults.num_events)
    output_file_path = kwargs.get("output_file", tasks.SimulationDefaults.output_file_path)
    random_seed = kwargs.get("random_seed", tasks.SimulationDefaults.random_seed)
    return tasks.simulate(source, sensor, readout, num_events, output_file_path,
                          random_seed, kwargs)
