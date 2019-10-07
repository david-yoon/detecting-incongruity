# coding: utf-8

"""
what    : AHDE data process
"""

import numpy as np
import random
import pickle
from tqdm import tqdm



class ProcessData:
    
    def __init__(self, is_test, params, data_path='../data/', evaluation_file_name=''):
        
        print 'IS_TEST = ' + str(is_test)
        
        self.params = params
        self.data_path = data_path
        
        self.is_test = is_test
        self.evaluation_file_name = evaluation_file_name
        
        self.voca = None
        self.pad_index = 0
        self.index2word = {}
        
        
        self.train_data = {}
        self.valid_data = {}
        self.test_data = {}
        
        
        self.train_set = []
        self.valid_set = []
        self.test_set = []
        
        self.load_data()
        
        self.create_data_set(self.train_data, self.train_set, 'TRAIN')
        self.create_data_set(self.valid_data, self.valid_set, 'VALID')
        self.create_data_set(self.test_data,  self.test_set, 'TEST' )

        
    def load_data(self):
        
        if self.params.IS_DEBUG :
            print 'load data : DEBUG mode'
            
            self.test_data['c'] = np.load(self.data_path + self.params.DATA_DEBUG_TITLE)
            self.test_data['r'] = np.load(self.data_path + self.params.DATA_DEBUG_BODY)
            self.test_data['y'] = np.load(self.data_path + self.params.DATA_DEBUG_LABEL)
            
            self.train_data['c'] = self.test_data['c']
            self.train_data['r'] = self.test_data['r']
            self.train_data['y'] = self.test_data['y']

            self.valid_data['c'] = self.test_data['c']
            self.valid_data['r'] = self.test_data['r']
            self.valid_data['y'] = self.test_data['y']
            
        elif self.is_test:
            print 'load data : TEST mode'
            
            self.test_data['c'] = np.load(self.data_path + self.params.DATA_TEST_TITLE)
            self.test_data['r'] = np.load(self.data_path + self.params.DATA_TEST_BODY)
            self.test_data['y'] = np.load(self.data_path + self.params.DATA_TEST_LABEL)
            
            self.train_data['c'] = self.test_data['c']
            self.train_data['r'] = self.test_data['r']
            self.train_data['y'] = self.test_data['y']

            self.valid_data['c'] = self.test_data['c']
            self.valid_data['r'] = self.test_data['r']
            self.valid_data['y'] = self.test_data['y']

        else:
            print 'load data : TRAIN mode'
            self.train_data['c'] = np.load(self.data_path + self.params.DATA_TRAIN_TITLE)
            self.train_data['r'] = np.load(self.data_path + self.params.DATA_TRAIN_BODY)
            self.train_data['y'] = np.load(self.data_path + self.params.DATA_TRAIN_LABEL)

            self.valid_data['c'] = np.load(self.data_path + self.params.DATA_DEV_TITLE)
            self.valid_data['r'] = np.load(self.data_path + self.params.DATA_DEV_BODY)
            self.valid_data['y'] = np.load(self.data_path + self.params.DATA_DEV_LABEL)
            
            self.test_data['c'] = np.load(self.data_path  + self.params.DATA_TEST_TITLE)
            self.test_data['r'] = np.load(self.data_path  + self.params.DATA_TEST_BODY)
            self.test_data['y'] = np.load(self.data_path  + self.params.DATA_TEST_LABEL)
            
        self.voca = pickle.load(open(self.data_path + self.params.VOCA_FILE_NAME, 'r') )
        
        print 'add pad index as : ' + str(self.voca[''])
        self.pad_index = self.voca['']
        
        for w in self.voca:
            self.index2word[self.voca[w]] = w
        
        print '[completed] load data'
        print 'voca size (include _PAD_, _UNK_): ' + str( len(self.voca) )
        
    """
    def func(input) :
        # <EOS> == 3
        return ' '.join(str(e) for e in input).split(' 3 ')[:-1]
    """    
        
    # create train set : 
    # source_ids : 
    # target_ids : 
    # convert to soucre, target, label
    def create_data_set(self, input_data, output_set, set_type):
        
        print 'create_datat_set (delimiter <EOP>): ', self.voca['<EOP>']
        
        if self.params.IS_DEBUG is not True:
            if self.is_test & (set_type != 'TEST') : return
        
        data_len = len(input_data['c'])
        
        for index in tqdm( xrange(data_len) ):
            
            delimiter = ' ' +  str(self.voca['<EOP>']) + ' '
            # last == padding
            turn =[x.strip() for x in (' '.join(str(e) for e in input_data['r'][index])).split(delimiter)[:-1] ]
            turn = [ x for x in turn if len(x) >1]
            
            tmp_ids = [x.split(' ') for x in turn]
            target_ids = []
            for sent in tmp_ids:
                target_ids.append( [ int(x) for x in sent]  )

            source_ids = input_data['c'][index]
            
            label = float(input_data['y'][index])
            
            output_set.append( [source_ids, target_ids, label] )
        
        print '[completed] create '  + set_type + ': ' +  str(len(output_set))
            
            

    def get_glove(self):
        
        print 'load.... pre-trained embedding : ' + self.params.GLOVE_FILE_NAME
        self.W_glove_init = np.load(open(self.data_path + self.params.GLOVE_FILE_NAME, 'r'))
        
        return self.W_glove_init
    
    
    """
        inputs: 
            data: 
            batch_size : 
            encoder_size : max encoder time step
            context_size : max context encoding time step
            encoderR_size : max decoder time step
            
            is_test : batch data generation in test case
            start_index : 
            target_index : 0, 1

        return:
            encoder_inputs : [batch x context_size, time_step]
            encoderR_inputs : [batch, time_step]
            encoder_seq :
            context_seq  :
            encoderR_seq :
            target_labels : label
    """
    def get_batch(self, data, batch_size, encoder_size, context_size, encoderR_size, is_test, start_index=0, target_index=1):

        encoder_inputs, encoderR_inputs, encoder_seq, context_seq, encoderR_seq, target_labels = [], [], [], [], [], []
        index = start_index
        
        # Get a random batch of encoder and encoderR inputs from data,
        # pad them if needed

        for _ in xrange(batch_size):

            if is_test is False:
                encoderR_input, list_encoder_input, target_label = random.choice(data)
            else:
                # overflow case
                if index > len(data)-1:
                    list_encoder_input = data[-1][1]
                    encoderR_input = data[-1][0]
                    target_label = data[-1][2]
                else:
                    list_encoder_input = data[index][1]
                    encoderR_input = data[index][0]
                    target_label = data[index][2]

                index = index +1
    
            list_len = len( list_encoder_input )
            tmp_encoder_inputs = []
            tmp_encoder_seq = []
            
            for en_input in list_encoder_input:
                encoder_pad = [self.pad_index] * (encoder_size - len( en_input ))
                tmp_encoder_inputs.append( (en_input + encoder_pad)[:encoder_size] )        
                tmp_encoder_seq.append( min( len( en_input ), encoder_size ) )    
            
            # add pad
            for i in xrange( context_size - list_len ):
                encoder_pad = [self.pad_index] * (encoder_size)
                tmp_encoder_inputs.append( encoder_pad )
                tmp_encoder_seq.append( 0 ) 

            encoder_inputs.extend( tmp_encoder_inputs[:context_size] )
            encoder_seq.extend( tmp_encoder_seq[:context_size] )
            
            context_seq.append( min(  len(list_encoder_input), context_size  ) )
            
            encoderR_length = np.where( encoderR_input==0 )[-1]
            if ( len(encoderR_length)==0 ) : encoderR_length = encoderR_size
            else : encoderR_length = encoderR_length[0]
            
            
            # encoderR inputs are padded
            encoderR_pad = [self.pad_index] * (encoderR_size - encoderR_length)
            encoderR_inputs.append( (encoderR_input.tolist() + encoderR_pad)[:encoderR_size])

            encoderR_seq.append( min(encoderR_length, encoderR_size) )

            # Target Label for batch
            target_labels.append( int(target_label) )
                                
                    
        return encoder_inputs, encoderR_inputs, encoder_seq, context_seq, encoderR_seq, np.reshape(target_labels, (batch_size, 1))
