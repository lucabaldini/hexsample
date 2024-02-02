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
        logger.info(f'About to generate {num_events} event(s)...')
        rate = num_events / self.total_time
        logger.info(f'Effective event rate: {rate}')
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
        It returns the list of times of the read evts and the number of dead evts.
        """
        last_read_event_idx = 0
        n_dead_events = 0
        read_events_times = [event_times[last_read_event_idx]]
        for idx, evt_time in enumerate(event_times):
            if idx == 0:
                continue
            elif (event_times[idx]-event_times[last_read_event_idx]) <= self.system_reading_time: 
            #the event cannot be read, counted as dead
                n_dead_events+=1
            else: #the event time has to be saved because has been read so it is passed to buffer
                read_events_times.append(evt_time) #event saved
                #changing the index of the last event read, is the new reference point
                last_read_event_idx = idx 

        t_arrival_read = np.array(read_events_times)
        return t_arrival_read, n_dead_events

class Buffer:

    """Buffer class constructor
    """

    def __init__(self, t_service: float, max_queue_lenght: int, n_processors: int=1):
        self._n_processors = n_processors
        self._t_service = t_service*self._n_processors #/mus
        self._max_queue_lenght = max_queue_lenght
        self.queue = [0]
        self.n_dead_evts = 0

    def __str__(self):
        """
        """
        formatted_print = f'Element of the class {self.__class__.__name__}\n' \
        'Features of the buffer:' \
        f'Service time : {self._t_service} mus\n' \
        f'Lenght of the buffer : {self._max_queue_lenght}\n' \
        f'Number of processors per buffer : {self._n_processors}'
        return formatted_print
    
    def rho(self, event_times):
        """This quantity is the relevant quantity in queuing theory. 
        It is computed as (rate of evts entering buffer)/(rate of evts exiting buffer).
        It has been kept the notation in Data Analysis Techniques for High-Energy Phisics
        (R. Fruhwirth et al.) - Section 1.3
        It is supposed that the event_times array has been already 'filtered' by
        readout after trigger.
        """
        lambd = 1/np.mean((np.diff(event_times)))
        mu = 1/self._t_service
        return lambd/mu
    
    def event_processing(self, event_times: np.array, total_time: float):
        """Given an array (or array-like) in input that contains event times, 
        this function transforms the array in an array that contains +1/-1
        that means evt in entrance/exit of buffer.
        After that, the function scans the array and computes for every step
        the queue lenght and updates the dead events count. 
        """
        #Resetting queue and dead evts (seems necessary but I do not get why)
        self.queue = [0]
        self.n_dead_evts = 0
        print(f'Queue has been reset: {self.queue}')
        #Creating the array with the right features for being processed:
        #We want an array of +1 and -1, that indicates the arrival and exit
        #of an event from the buffer. 
        service_times = np.arange(self._t_service, max(max(event_times),total_time), self._t_service) #ask if it is fine
        times = np.concatenate((event_times, service_times))
        idxes = np.argsort(times)
        times = times[idxes]
        w = np.concatenate((np.full(event_times.shape, 1), np.full(service_times.shape, -1)))
        w = w[idxes] #this is the array to give as input to the buffer.

        for evt in w:
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
    
class Readout: #probably useless..l let's think about it
    #maybe not, it summarizes all the important quantities of the readout. 
    def __init__(self, triggerreadout: TriggerReadout, buffer: Buffer):
        self.triggerreadout = triggerreadout
        self.buffer = buffer

    def effective_rate_entering_buffer(self, event_times: np.array) -> float:
        event_times_to_buffer = self.triggerreadout.readout_after_trigger(event_times)[0]
        return 1/np.mean((np.diff(event_times_to_buffer)))

    def deadevts_before_buffer(self, event_times: np.array) -> float:
        return self.triggerreadout.readout_after_trigger(event_times)[1]
    
    def fraction_deadevts_before_buffer(self, event_times: np.array) -> float:
        return (self.triggerreadout.readout_after_trigger(event_times)[1])/(len(event_times))
    
    def deadevts_after_buffer(self, event_times: np.array, total_time: float) -> float:
        event_times_to_buffer = self.triggerreadout.readout_after_trigger(event_times)[0]
        print(len(event_times_to_buffer))
        self.buffer.event_processing(event_times_to_buffer, total_time)
        return self.buffer.n_dead_events
    
    def fraction_deadevts_after_buffer(self, event_times: np.array, total_time: float) -> float:
        event_times_to_buffer = self.triggerreadout.readout_after_trigger(event_times)[0]
        self.buffer.event_processing(event_times_to_buffer, total_time)
        return (self.buffer.n_dead_evts)/(len(event_times_to_buffer))
    
    def fraction_total_dead_evts(self, event_times: np.array, total_time: float) -> float:
        event_times_to_buffer = self.triggerreadout.readout_after_trigger(event_times)[0]
        self.buffer.event_processing(event_times_to_buffer, total_time)
        return (len(event_times)-(self.deadevts_after_buffer(event_times)+\
                                  self.deadevts_before_buffer(event_times)))/len(event_times)
    
if __name__ == "__main__":
    #Defining the features of the system
    total_time = 1 #s
    event_size = 116*2 #bit/evt (116 bits is the expected event size, the *2 is for conservativeness)
    reading_time = 4e-6 #s
    readout_thruput = (500.e6)/8 #bits/s
    service_time = event_size / readout_thruput #s
    #Defining the grid of values that we want to simulate
    average_rate = [1e3, 10e3, 50e3, 70e3, 100e3, 150e3, 200e3, 220e3, 250e3, 270e3, 300e3, 350e3, 400e3, 450e3, 500e3] #Hz
    max_buffer_lenght = [1, 2, 4, 8]
    #Defining the Readout objects containing the specifics
    readouts = []
    for buff_lenght in max_buffer_lenght:
        #In the following list there is a Readout object for buffer lenght
        readouts.append(Readout(TriggerReadout(reading_time), Buffer(service_time, buff_lenght)))
    for readout in readouts:
        print(f'Working with readout having lenght={readout.buffer._max_queue_lenght}...')
        logger.info(f'Service time: {readout.buffer._t_service} s')
        fdead_events_bb = [] #dead evts before buffer
        fdead_events_ab = [] #dead evts after buffer
        effective_rates = []
        for rate in average_rate:
            print(f'Considering rate {rate}...')
            eventgenerator = EventGenerator(rate, total_time)
            t_events = eventgenerator.event_times_creation()
            effective_rates.append(readout.effective_rate_entering_buffer(t_events))
            fdead_events_bb.append(readout.fraction_deadevts_before_buffer(t_events))
            fdead_events_ab.append(readout.fraction_deadevts_after_buffer(t_events, total_time))
            rho = readout.buffer.rho(t_events)
        plt.figure('Fraction of dead events after 1st stage of readout')
        plt.grid(True)
        plt.title('Fraction of dead events after 1st stage of readout')
        plt.plot(average_rate, fdead_events_bb, label=fr'$L_{{buffer}}$ = {readout.buffer._max_queue_lenght}')
        plt.xlabel(r'Events rate $\lambda$ [Hz]')
        plt.ylabel(r'$\frac{\text{dead events}}{\text{all events}}$')

        plt.figure('Fraction of dead events after buffer stage')
        plt.grid(True)
        plt.title('Fraction of dead events after buffer stage')
        plt.plot(average_rate, fdead_events_ab, label=fr'$L_{{buffer}}$ = {readout.buffer._max_queue_lenght}, $\rho$={rho:.2f}')
        #plt.axvline(1/readout.buffer._t_service, linestyle='dashed', color='r', label=r'Buffer service rate $\mu$ [Hz]')
        plt.xlabel(r'Events rate $\lambda$ [Hz]')
        plt.ylabel(r'$\frac{\text{dead events}}{\text{all events}}$')
        plt.legend()

        plt.figure('Effective rates entering buffer')
        plt.title('Effective event rate entering buffer as a function of real event rate')
        plt.grid(True)
        plt.plot(average_rate, effective_rates)
        plt.xlabel(r'Events rate $\lambda$ [Hz]')
        plt.ylabel('Effective events rate [Hz]')

    plt.axhline(1/readout.buffer._t_service, linestyle='dashed', color='r', label=r'Buffer service rate $\mu$ [Hz]')
    plt.legend()
    plt.show()