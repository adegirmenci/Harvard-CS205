'''Driver for P6.py'''
__author__ = 'Alperen Degirmenci'

from P6 import mrP6
from collections import defaultdict
import logging
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print '\n-> Starting program...\n'
    
    # Create a list of books
    bookList = []
    bookList.append('CanterburyTales.txt')
    bookList.append('Frankenstein.txt')
    bookList.append('KingJamesBible.txt')
    bookList.append('ParadiseLost.txt')
    bookList.append('PrideAndPrejudice.txt')
    bookList.append('TaleOfTwoCities.txt')
    bookList.append('WutheringHeights.txt')
    numBooks = len(bookList)
    args = []
 
    # Set logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stderr))
    
    # Create a list for storing defaultdicts yielded by the reducer
    results = []

    # Instantiate MRJob child class mrP4
    myP6 = mrP6(args)
    
    # Loops mrP4 until we have convergence
    for book in bookList:
        print '\n-> Analyzing:', book, '\n'
        # Redirect book contents to MR job
        with open(book, 'r') as inF:
            myP6.stdin = inF.readlines()
        # Create a defaultDict for storing results
        resultDict = defaultdict(int)
        # Run myP6 using a runner
        with myP6.make_runner() as runner: # make a runner
            runner.run() # run job
            for line in runner.stream_output():
                key, value = myP6.parse_output_line(line)
                resultDict[key]+=value
        results.append(resultDict)

    # Construct array of x-values
    x = np.arange(0,26,1)

    # Make a list of letters in the alphabet
    letters = []
    for i in x:
        letters.append(''.join(chr(i+65)))

    # Count number of letters in each book for normalization at graphing stage
    letterCount = []
    for n in xrange(0, len(results)):
        cnt = 0
        for i in xrange(0, len(letters)):
            cnt += results[n][letters[i]]
        letterCount.append(cnt)
    
    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lineColors = ['-r', '-b', '-g', '-c', '-y', '-m', '-k']
    # Get y-axis values
    y = np.zeros([len(letters), len(results)])
    for n in xrange(0, len(results)):
        for c in xrange(0,len(letters)):
            y[c, n] = float(results[n][letters[c]]*100)/float(letterCount[n])
        # Plot
        plt.plot(x.T, y[:,n], lineColors[n], label=bookList[n])

    # Axis adjustments
    plt.xlabel('Letters')
    plt.ylabel('Percentage of occurences')
    plt.title('Trend Analysis for Letters in Books')
    plt.legend()
    ax.set_xticks(np.arange(0,len(letters)))
    ax.set_xticklabels(letters)
    plt.show()

    print '\-> Analysis complete!\n'
