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


class Buffer:

    """Buffer class constructor
    FOR NOW, WORKS WITH SINGLE PROCESSOR ONLY, MULTIPROCESSOR NEEDS TO BE IMPLEMENTED
    """

    def __init__(self, t_service: float, max_queue_lenght: int, n_processors: int=1, queue: list=[0], n_dead_evts: int=0):
        self._t_service = t_service #/mus
        self._max_queue_lenght = max_queue_lenght
        self._n_processors = n_processors
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
        """Given an array (or array-like) in input that contains +1 when 1 evt 
        (could be plural for the function but for now it is not useful), enters
        the buffer and -1 when an event exits the buffer, fills a list containing
        the queue lenghts for every steps and the number of dead events.
        """
        for evt in events:
            if evt > 0: #events in input
                if self.queue[-1] < self._max_queue_lenght: #all ok
                    self.queue.append(self.queue[-1]+evt)
                else: #full queue, saturating the lenght and counting dead evts
                    self.queue.append(self.queue[-1])
                    self.n_dead_evts+=evt
            else: #nothing happened or evts in output
                if self.queue[-1] == 0:
                    self.queue.append(0)
                else:
                    self.queue.append(self.queue[-1]+evt)
        return

def test_buffer():
    """
    """

    total_time = 0.0001 #s
    average_rate = 250000. #Hz
    event_size = 1000 #bits
    readout_thruput = 100.e6 #bits/s
    service_time = event_size / readout_thruput #s

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
    t_service = np.arange(service_time, total_time, service_time)
    #Creating the array with the right features for being processed:
    #We want an array of +1 and -1, that indicates the arrival and exit
    #of an event from the buffer. 
    t = np.concatenate((t_arrival, t_service))
    w = np.concatenate((np.full(t_arrival.shape, 1), np.full(t_service.shape, -1)))
    i = np.argsort(t)
    t = t[i]
    w = w[i] #this is the array to give as input to the buffer.
    '''
    plt.figure('Buffer timeline')
    plt.bar(t_arrival, 1., width=1.e-8, color='black')
    plt.bar(t_service, -1., width=1.e-8, color='red')
    plt.plot(t, np.cumsum(w))
    setup_gca(xmin=0., xmax=total_time, xlabel='Time [s]', ymin=-2., ymax=20.)
    '''
    #Instantiating the buffer
    buffer = Buffer(t_service=service_time, max_queue_lenght=4)
    buffer.event_processing(w)
    queue_lenght = buffer.queue
    dead_events = buffer.n_dead_evts
    print(buffer.queue)
    print(f'Number of dead events: {buffer.n_dead_evts}')
    print(len(t))
    t=np.insert(t,0,0)
    print(len(t))
    print(len(queue_lenght))

    plt.figure('Buffer timeline')
    plt.bar(t_arrival, 1., width=1.e-8, color='black')
    plt.bar(t_service, -1., width=1.e-8, color='red')
    plt.plot(t, queue_lenght)
    setup_gca(xmin=0., xmax=total_time, xlabel='Time [s]', ymin=-2., ymax=20.)


if __name__ == '__main__':
    test_buffer()
    plt.show()
