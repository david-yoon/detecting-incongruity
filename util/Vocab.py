# coding: utf-8
'''
what: vocabulary class for word2index, index2word & sentence -> index list
'''

import numpy as np
import operator

class Vocab:
    
    
    def __init__(self, dic):        
        self.dic = dic
        self.index = []
        self._create_index()
        
        
    def _create_index(self):
        sorted_voca = sorted(self.dic.items(), key=operator.itemgetter(1))
        for word, num in sorted_voca:
            self.index.append( word )
            
            
    def find_index(self, word):
        if self.dic.has_key(word):
            return self.dic[word]
        else:
            return self.dic['_UNK_']  

        
    """
    input: list( index )
    output: str (sentnece)
    """
    def index2sent(self, index):
        
        if len(index) == 1:
            return self.index[index[0]]
        else:
            return ' '.join( [ self.index[int(x)] for x in index ] )      
    
    
    """
    input: list ( word )
    output: list ( index )
    """
    def word2index(self, word):
        
        if len(word) > 1:
            return [ int(self.find_index(x)) for x in word ]
        else :
            return [ int(self.find_index(word[0])) ]
        

    # sentence --> list of index
    def __call__(self, line):        
        """
        if type(line) is np.ndarray:
            return " ".join([self.index2word[word] for word in line])
        
        if type(line) is list:
            if len(line) > 0:
                if line[0] is int:
                    return " ".join([self.index2word[word] for word in line])
            indices = np.zeros(len(line), dtype=np.int32)
        else:
            line = line.split(" ")
            indices = np.zeros(len(line), dtype=np.int32)
        
        for i, word in enumerate(line):
            indices[i] = self.word2index.get(word, self.unknown)
            
        return indices
        """
    
    @property
    def size(self):
        return len(self.dic)
    
    def __len__(self):
        return len(self.dic)  
