# Copyright (C) 2022--2023 luca.baldini@pi.infn.it
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

"""Buffer class for chip readout and related quantities
"""

import numpy as np

from hexsample import rng
from hexsample import logger
from hexsample.plot import plt, setup_gca


    
def event_generator(average_rate: float, total_time: float) -> np.array:
    """The following function generates the number of events and then the
    time of arrival of each. It returns the array containing the times of
    arrival of the events.

    Arguments
    ---------
    average_rate : float
        Average rate of the events to generate in Hz.
    total_time : float 
        Maximum time for an event to occur in s (starting from 0 s).

    Return
    ------
    t_arrival : np.array
        Array containing the times of the events. Its lenght varies following
        a Poisson distribution having as mean average_rate * total_time
    """
    #Generating the number of events with a Poissonian distribution
    # and then (with the true rate), generating times of arrival of
    # every event. 
    rng.initialize()
    num_events = rng.generator.poisson(average_rate * total_time)
    logger.info(f'About to generate {num_events} event(s)...')
    rate = num_events / total_time
    logger.info(f'Effective event rate: {rate}')
    #time deltas between events are generated with an exponential and then
    #summed over in order to obtain absolute times in s, starting from 0 s.
    dt_events = rng.generator.exponential(1. / rate, size=num_events)
    t_events = np.cumsum(dt_events)
    return t_events

def buffers_shifting_to_pc(t_start: float, t_stop: float,\
                            len_queue1: float,len_queue2: float, max_len1: float, max_len2: float,
                            t_service1: np.array,t_service2: np.array):
    """This function processes the events inside two buffers queue in cascade,
    having service times inside the simulation fixed outside the function and 
    given as input.
    The processing in computed between [t_start, t_stop], trying to pass events
    from the first buffer to the second one and events from second buffer to 
    the 'outside' (service PC). 
    """

    return 