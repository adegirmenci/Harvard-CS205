"""P3.py: Solves anagrams. Uses in-mapper combining"""
__author__ = 'Alperen Degirmenci'

from mrjob.job import MRJob
from collections import defaultdict

class mrAnagram(MRJob):
    def __init__(self, *args, **kwargs):
        super(mrAnagram, self).__init__(*args,**kwargs)
        self.localAnags = defaultdict(list)

    def mapper(self,key,line):
        # each line is a word
        line = line.split('\n') # get rid of the return character
        word = line[0]
        # sort characters in the word
        w = list(word)  # make it into a list for sorting
        w.sort()        # sort
        sorted_w = ''.join(w) # convert back to str
        if sorted_w == word:  # the sorted word is the same as the word
            self.localAnags[sorted_w] # don't need to add it to the list
        else:
            self.localAnags[sorted_w].append(word)

    def mapper_final(self):
        # emit the contents of the dict
        for (sorted_w, words) in self.localAnags.iteritems():
            yield sorted_w, words

    def reducer(self,sorted_w,words):
        anags = sum(words, []) # combine lists
        yield sorted_w, (len(anags), anags)

if __name__ == '__main__':
    mrAnagram.run()
