'''CS205 - HW1 - P5 - Monte Carlo Simulation with In-Mapper Combining'''
__author__ = 'Alperen Degirmenci'

from mrjob.job import MRJob
from math import sqrt
import random

class mrP5(MRJob):
    def __init__(self,*args,**kwargs):
        super(mrP5, self).__init__(*args,**kwargs)
        self.num = 0
        self.denom = 0
        self.numKey = 'numerator'
        self.denomKey = 'denominator'

    def mapper(self,key,line):
        # seed a random number generator per input
        random.seed(int(line))
        num = 0
        denom = 0
        for n in xrange(0,1000):
            x = random.random()
            y = random.random()
            # check if pair is under curve
            if sqrt(x**2 + y**2) <= 1.0:
                num += 1    
            denom += 1
        # Writing to self. variables at the end makes the program run faster. I have observed around 25% speedup
        self.num += num
        self.denom += denom

    def mapper_final(self):
        yield self.numKey, self.num
        yield self.denomKey, self.denom

    def reducer(self,word,occurrences):
        yield word, sum(occurrences)

if __name__ == '__main__':
    mrP5.run()
