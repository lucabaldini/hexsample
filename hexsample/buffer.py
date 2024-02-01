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

class EventGenerator:
    """Event generator constructor
    """
    def __init__(self, average_rate: float, total_time: float):
        self.average_rate = average_rate
        self.total_time = total_time
    
    def event_times_creation(self):
        """The following function generates the number of events and then the
        time of arrival of each. It returns the array containing the times of
        arrival of the events.
        """
        #Generating the number of events with a Poissonian distribution
        # and then (with the true rate), generating times of arrival of
        # every event. 
        rng.initialize()
        num_events = rng.generator.poisson(self.average_rate * self.total_time)
        rate = num_events / self.total_time
        #logger.info(f'Service time: {self.service_time} s')
        #logger.info(f'About to generate {num_events} event(s)...')
        #delta between events are generated with an exponential and then 
        #summed over in order to obtain absolute times in s.
        dt_arrival = rng.generator.exponential(1. / rate, size=num_events)
        t_arrival = np.cumsum(dt_arrival)
        return t_arrival
    
class TriggerReadout:
    def __init__(self, system_reading_time: float):
        self.system_reading_time = system_reading_time
        
    def readout_after_trigger(self, event_times: np.array):
        """This function takes as input the event times. considering the system dead
        time, it cancels the times of the unread events and counts how many they are.
        It returns the list of times and the number of dead events.
        """
        last_read_event_idx = 0
        n_dead_events = 0
        read_events_times = [event_times[last_read_event_idx]]
        for idx, evt_time in enumerate(event_times[1:]):
            if (event_times[idx]-event_times[last_read_event_idx]) <= self.system_reading_time: #the event cannot be read, counted as dead
                n_dead_events+=1
            else: #the event time has to be saved because has been read so it is passed to buffer
                read_events_times.append(evt_time) #event saved
                last_read_event_idx = idx #changing the index of the last event read, is the new reference point
        t_arrival_read = np.array(read_events_times)

        return t_arrival_read, n_dead_events


class Buffer:

    """Buffer class constructor
    """

    def __init__(self, t_service: float, max_queue_lenght: int, n_processors: int=1, queue: list=[0], n_dead_evts: int=0):
        self._n_processors = n_processors
        self._t_service = t_service*self._n_processors #/mus
        self._max_queue_lenght = max_queue_lenght
        self.queue = queue
        self.n_dead_evts = n_dead_evts

    def __str__(self):
        """
        """
        formatted_print = f'Element of the class {self.__class__.__name__}\n' \
        'Features of the buffer:' \
        f'Service time : {self._t_service} mus\n' \
        f'Lenght of the buffer : {self._max_queue_lenght}\n' \
        f'Number of processors per buffer : {self._n_processors}'
        return formatted_print
    
    def event_processing(self, events: np.array):
        """Given an array (or array-like) in input that contains event times, 
        this function transforms the array in an array that contains +1/-1
        that means evt in entrance/exit of buffer.
        After that, the function scans the array and computes for every step
        the queue lenght and updates the dead events count. 
        """
        
        for evt in events:
            if evt > 0: #events in input
                if self.queue[-1] < self._max_queue_lenght: #all ok
                    self.queue.append(self.queue[-1]+evt)
                elif self.queue[-1] == self._max_queue_lenght: #full queue, saturating the lenght and counting dead evts
                    self.queue.append(self.queue[-1])
                    self.n_dead_evts+=evt
            else: #nothing happened or evts in output
                if self.queue[-1] == 0:
                    self.queue.append(0)
                else:
                    self.queue.append(self.queue[-1]+evt)
        return

def readout_after_trigger(event_times: np.array, system_deadtime: float):
    """This function takes as input the event times. considering the system dead
    time, it cancels the times of the unread events and counts how many they are.
    It returns the list of times and the number of dead events.
    """
    last_read_event_idx = 0
    n_dead_events = 0
    read_events_times = [event_times[last_read_event_idx]]
    for idx, evt_time in enumerate(event_times[1:]):
        if (event_times[idx]-event_times[last_read_event_idx]) <= system_deadtime: #the event cannot be read, counted as dead
            n_dead_events+=1
        else: #the event time has to be saved because has been read so it is passed to buffer
            read_events_times.append(evt_time) #event saved
            last_read_event_idx = idx #changing the index of the last event read, is the new reference point
    t_arrival_read = np.array(read_events_times)

    return t_arrival_read, n_dead_events
    

def test_buffer(total_time: float, average_rate: float, reading_time: float, service_time: float, buffer_lenght: int=1):
    """This is a test for a single buffer readout. 
    """
    
    #Generating the number of events with a Poissonian distribution
    # and then (with the true rate), generating times of arrival of
    # every event. 
    rng.initialize()
    num_events = rng.generator.poisson(average_rate * total_time)
    rate = num_events / total_time
    logger.info(f'Service time: {service_time} s')
    logger.info(f'About to generate {num_events} event(s)...')
    #delta between events are generated with an exponential and then 
    #summed over in order to obtain absolute times in s.
    dt_arrival = rng.generator.exponential(1. / rate, size=num_events)
    t_arrival = np.cumsum(dt_arrival)
    #After creating time events, those need to be cut if their lag is less than reading time
    t_arrival_read, num_dead_events = readout_after_trigger(t_arrival, reading_time)
    dt_arrival_read = np.diff(t_arrival_read, prepend=0)
    average_rate_entering_buffer = 1/(np.mean(dt_arrival_read))
    print(f'THE EFFECTIVE RATE ENTERING BUFFER IS: {average_rate_entering_buffer}')
    f_accepted_events_after_trigger = (len(t_arrival_read))/len(t_arrival)
    #Creating the set of service times of the buffer
    t_service = np.arange(service_time, max(max(t_arrival),total_time), service_time) #ask if it is fine

    #Creating the array with the right features for being processed:
    #We want an array of +1 and -1, that indicates the arrival and exit
    #of an event from the buffer. 
    t = np.concatenate((t_arrival_read, t_service))
    i = np.argsort(t)
    t = t[i]
    w = np.concatenate((np.full(t_arrival_read.shape, 1), np.full(t_service.shape, -1)))
    w = w[i] #this is the array to give as input to the buffer.
    '''
    plt.figure('Buffer timeline')
    plt.bar(t_arrival, 1., width=1.e-8, color='black')
    plt.bar(t_service, -1., width=1.e-8, color='red')
    plt.plot(t, np.cumsum(w))
    setup_gca(xmin=0., xmax=total_time, xlabel='Time [s]', ymin=-2., ymax=20.)
    '''
    #Instantiating the buffer
    buffer = Buffer(t_service=service_time, max_queue_lenght=buffer_lenght, n_dead_evts=0)
    buffer.queue=[0] #ask why it is necessary
    buffer.event_processing(w)
    queue_lenght = len(buffer.queue)
    dead_events = buffer.n_dead_evts
    f_accepted_events = (len(t_arrival_read) - dead_events)/len(t_arrival_read)
    print(f'Number of dead events for {average_rate} Hz rate with buffer lenght {buffer._max_queue_lenght}: {dead_events}')
    
    t=np.insert(t,0,0) #this is needed for having len(t) == len(queue) (t lacks of the 0 point)
    '''
    Summary plots
    plt.figure('Buffer timeline')
    plt.grid()
    plt.bar(t_arrival, 1., width=1.e-8, color='black')
    plt.bar(t_service, -1., width=1.e-8, color='red')
    plt.plot(t, buffer.queue)
    setup_gca(xmin=0., xmax=total_time, xlabel='Time [s]', ymin=-2., ymax=10.)
    '''
    return f_accepted_events_after_trigger, f_accepted_events


if __name__ == '__main__':
    #Defining the features of the system
    total_time = 10 #s
    event_size = 116*2 #bit/evt (116 bits is the expected event size, the *2 is for conservativeness)
    #reading_time = 4e-6 #s
    reading_time=4e-6 #s
    #readout_thruput = (50.e6)/8 #bits/s
    readout_thruput = (500.e6)/8 #bits/s
    service_time = event_size / readout_thruput #s
    #Defining the grid of values that we want to simulate
    average_rate = [1e3, 10e3, 50e3, 70e3, 100e3, 150e3, 200e3] #Hz
    max_buffer_lenght = [1, 2, 4, 8]
    plt.figure()
    for buffer_lenght in max_buffer_lenght:
        fractions_before_buffer = []
        fractions_after_buffer = []
        for rate in average_rate:
            f1, f2 = test_buffer(total_time, rate, reading_time, service_time, buffer_lenght)
            fractions_before_buffer.append(f1)
            fractions_after_buffer.append(f2)
        plt.errorbar(average_rate, fractions_after_buffer, marker='.', label=f'Buffer lenght = {buffer_lenght}')
    

    plt.grid()
    plt.xlabel('Average event rate')
    plt.ylabel('Fraction of events elaborated by buffer')
    plt.legend()
    plt.show()
