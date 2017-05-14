'''Driver for P4.py'''
__author__ = 'Alperen Degirmenci'

from P4 import mrP4
from mrjob.protocol import JSONProtocol
import logging
import sys
import time

if __name__ == '__main__':
    print '\n->Strating program...'
    
    # Set IO variables
    inputFileName = './graph.txt'
    outputFileName = './output.txt'
    args = []
    # Announce IO varaibles
    print '\n-> Using Input File:', inputFileName
    print '-> Using Output File:', outputFileName

    # Set logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stderr))
    
    # Instantiate MRJob child class mrP4
    myP4 = mrP4(args)
    # Open input file, feed into mrP4
    inF = open(inputFileName, 'r')
    myP4.stdin = inF.readlines()
    inF.close()
    
    # Variables for keeping track of convergence
    converged = False # keeps track of convergence
    ctrValue = 0 # value of the counter (number of better paths found)
    iteration = 0 # number of iterations
    # Loops mrP4 until we have convergence
    while not converged:
        with myP4.make_runner() as runner: # make a runner
            runner.run() # run job
            nextIn = open(outputFileName, 'w')
            for line in runner.stream_output():
                key, value = myP4.parse_output_line(line)
                nextIn.write(JSONProtocol.write(key, value) + '\n')
                print '-> Output of MR Job is:', key, value
            nextIn.close()
            iteration += 1 # update number of iterations
            # Get counter
            ctr = runner.counters()
            ctrValue = ctr[0]['reducer']['better found'] # Extract counter value
            if (ctrValue == 0):
                converged = True # We have convergence
        # Get previous run's values
        with open(outputFileName, 'r') as nextIn:
            myP4.stdin = nextIn.readlines()

    # Output file reorganization
    s = []
    with open(outputFileName, 'r') as f:
        for line in f:
            s.append(line)
    s.sort()
    with open(outputFileName, 'w') as f:
        for i in s:
            f.write(i)
        result = '\n-> Number of iterations was : ' + str(iteration)
        f.write(result)

    print '\n-> Results are written to the output file:', outputFileName, '\n'
    print result, '\n'
