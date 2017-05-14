'''Breadth-first search'''
__author__ = 'Alperen Degirmenci'

from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol
import sys

class mrP4(MRJob):
    INPUT_PROTOCOL = JSONProtocol

    # for each connected node, emit distance
    def mapper(self, nid, info):
        d = info[0]
        yield(nid, info) # emit node structure
        for n in info[1:]:
            yield n[0], (d + n[1]) # emit connected nodes

    # select minimum distance
    def reducer(self, nid, dList):
        dmin = 999 # initial min distance
        M = [] # empty node
        for d in dList: # d can be a node structure or a distance
            if not isinstance(d, int): # d is a node structure
                M = d # M is the node structure
                if M[0] < dmin: #M[0] is the distance to root from M
                    dmin = M[0] #min dist becomes the current distance
            elif d < dmin: # d is the distance
                dmin = d #min dist is current distance
        if dmin < M[0]:
            # tell P4Driver that the reducer has found a better distance
            self.increment_counter('reducer', 'better found', 1)
        else: # didn't find a better distance
            self.increment_counter('reducer', 'better found', 0)
        M[0] = dmin # update distance from M to root 
        yield nid, M # emit structure

if __name__ == '__main__':
    mrP4.run()

