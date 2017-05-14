# Written by Alperen Degirmenci on 9/22/12
# CS205 - HW1 - Implementation with In-mapper Combining

from mrjob.job import MRJob
import time
import sys

class mrSumInMap(MRJob):
    def __init__(self,*args,**kwargs):
        super(mrSumInMap, self).__init__(*args,**kwargs)
        self.sum = 0.0
        self.stps = 100001
        self.h = 1.0/self.stps
        self.word = 'pi/4'

    def mapper(self,key,line):
        #each line is a y value
        self.sum += float(line)

    def mapper_final(self):
        yield self.word, self.sum*self.h

    def reducer(self,word,occurrences):
        yield word, sum(occurrences)

if __name__ == '__main__':
    #t1 = time.time()
    mrSumInMap.run()
    #t2 = time.time()
    #result = '\n-> Time: ' + str(t2-t1) + ' seconds\n'
    #sys.stderr.write(result)
