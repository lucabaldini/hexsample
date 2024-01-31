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

class Buffer:
    """Buffer class constructor
    """
    def __init__(self, T_SERVICE: float, MAX_LENGHT: int, N_PROCESSORS: int, queue_lenght: int=0):
        self.T_SERVICE = T_SERVICE #/mus
        self.MAX_LENGHT = MAX_LENGHT
        self.N_PROCESSORS = N_PROCESSORS
        self.queue_lenght = queue_lenght
    
    def __str__(self):
        formatted_print=f'Element of the class {self.__class__.__name__}\n \
        Features of the buffer:
        Service time : {self.T_SERVICE} mus\n \
        Lenght of the buffer : {self.MAX_LENGHT}\n \
        Number of processors per buffer : {self.N_PROCESSORS}'
        return formatted_print

    def store_or_reject(self, n_input_evts: int=1) -> int:
        """This function checks the lenght of the queue and decides if an input event
        is stored or discarded due to full queue. 
        It updates the lenght of the queue if needed and returns the number
        of discarded events in order keep track of the total 'dead' evts. 
        """
        empty_queue_spaces = (self.MAX_LENGHT-self.queue_lenght)
        if n_input_evts <= empty_queue_spaces:
            #Buffer can accept all new events, queue_lenght is updated.
            #all new evts can be accepted
            self.queue_lenght+=n_input_evts
            return 0
        else:
            #filling all empty spaces and return the number unaccepted evts
            self.queue_lenght+=(self.MAX_LENGHT-self.queue_lenght)
            return n_input_evts - (self.MAX_LENGHT-self.queue_lenght)
        
    def queue_diminishment(self, n_served_evts: int=1):
        if self.queue_lenght - n_served_evts < 0:
            self.queue_lenght=0
            return
        else: 
            self.queue_lenght -= n_served_evts
            return
