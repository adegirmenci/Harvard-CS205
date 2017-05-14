# Written by Alperen Degirmenci on 9/22/12
# CS205 - HW1 - Implementation without In-mapper Combining

from mrjob.job import MRJob
import time
import sys

class mrSum(MRJob):
    def __init__(self, *args, **kwargs):
        super(mrSum, self).__init__(*args,**kwargs)
        self.stps = 100001
        self.h = 1.0/self.stps

    def mapper(self,key,line):
        #each line is a y value
        y = float(line)*self.h
        yield 'pi/4', y

    def reducer(self,word,occurrences):
        yield word, sum(occurrences)

if __name__ == '__main__':
    #t1 = time.time()
    mrSum.run()
    #t2 = time.time()
    #result = '\n-> Time: ' + str(t2-t1) + ' seconds\n'
    #sys.stderr.write(result)
