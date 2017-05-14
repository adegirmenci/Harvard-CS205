'''Culturomics - P6.py'''
__author__ = 'Alperen Degirmenci'

from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol
from collections import defaultdict

class mrP6(MRJob):
    def __init__(self, *args, **kwargs):
        super(mrP6, self).__init__(*args, **kwargs)
        self.localDict = defaultdict(int)

    def mapper(self, key, line):
        lin = list(line.upper())
        for ch in lin:
            if ch.isupper():
                self.localDict[ch]+=1

    def mapper_final(self):
        for (ch, count) in self.localDict.iteritems():
            yield ch, count

    def reducer(self, ch, occurrences):
        yield ch, sum(occurrences)

if __name__ == '__main__':
    mrP6.run()
