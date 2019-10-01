# coding: utf-8
'''
what: vocabulary class for word2index, index2word & sentence -> index list
'''

import numpy as np
import operator

class Vocab:
    
    def __init__(self, list_voca):        
        self.list_voca = list_voca
        self.index = {}
        for word in list_voca:
            self.index[word] = len(self.index)

            
    def find_index(self, word):
        if word in self.index:
            return self.index[word]
        else:
            return self.index['_UNK_']

        
    """
    input: list_index
    output: str (sentnece)
    """
    def index2sent(self, list_index):
        
        if len(list_index) > 1:
            return ' '.join( [ str(self.list_voca[int(x)]) for x in list_index ] )      
        else:
            return self.list_voca[0]
            
    
    """
    input: list_word
    output: list_index
    """
    def word2index(self, list_word):
        
        if len(list_word) > 1:
            return [ int(self.find_index(x)) for x in list_word ]
        else :
            return [ int(self.find_index('_PAD_')) ]
        

        
    # sentnece to index
    def __call__(self, sent):
        self.word2index(sent)
        
    
    @property
    def size(self):
        return len(self.list_voca)
    
    def __len__(self):
        return len(self.list_voca)  
