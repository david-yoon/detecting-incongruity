# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import DropoutWrapper, ResidualWrapper


# cell instance
def gru_cell(hidden_dim):
    return tf.contrib.rnn.GRUCell(num_units=hidden_dim)
    
# cell instance with drop-out wrapper applied
def drop_out_cell(hidden_dim=None, dr_in=1.0, dr_out=1.0, is_residual=False):
    if is_residual:
        print ('residual connection')
        return tf.contrib.rnn.ResidualWrapper( tf.contrib.rnn.DropoutWrapper(gru_cell(hidden_dim), input_keep_prob=dr_in, output_keep_prob=dr_out) )
    else: 
        return tf.contrib.rnn.DropoutWrapper(gru_cell(hidden_dim), input_keep_prob=dr_in, output_keep_prob=dr_out)
    
   
def add_GRU(inputs, inputs_len, cell = None, cell_fn = tf.contrib.rnn.GRUCell, hidden_dim=100, layers = 1, scope = "add_GRU", output = 0, is_training = True, reuse = None, dr_input_keep_prob=1.0, dr_output_keep_prob=1.0, is_bidir=False, is_bw_reversed=False, is_residual=False):
    '''
    Bidirectional recurrent neural network with GRU cells.

    Args:
        inputs:     rnn input of shape (batch_size, timestep, dim)
        inputs_len: rnn input_len of shape (batch_size, )
        cell:       rnn cell of type RNN_Cell.
        output:     [ batch, step, dim (fw;bw) ], [ batch, dim (fw;bw) ]
    '''
    with tf.variable_scope(name_or_scope=scope, reuse=reuse, initializer=tf.orthogonal_initializer()):
        
        if cell is not None:
            (cell_fw, cell_bw) = cell
        
        else:
            cell_fw = MultiRNNCell(
                [ drop_out_cell(
                    hidden_dim=hidden_dim,
                    dr_in=dr_input_keep_prob,
                    dr_out=dr_output_keep_prob,
                    is_residual=is_residual
                    ) for i in range(layers)
                ]
            )
                
                
                
            if is_bidir :
                print ("[layers] bidir case")
                cell_bw = MultiRNNCell(
                    [ drop_out_cell( 
                        hidden_dim=hidden_dim,
                        dr_in=dr_input_keep_prob,
                        dr_out=dr_output_keep_prob,
                        is_residual=is_residual
                    ) for i in range(layers)
                    ]
                )

        if is_bidir :
            # output : [ (fw,bw), batch, step, dim ]
            # status : [ (fw,bw), layer, seq, embed_dim ]
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                                                            cell_fw = cell_fw,
                                                            cell_bw = cell_bw,
                                                            inputs = inputs,
                                                            sequence_length = inputs_len,
                                                            dtype = tf.float32,
                                                            scope = scope,
                                                            time_major=False
                                                        )
            if is_bw_reversed :
                fw = outputs[0]
                bw = tf.reverse_sequence(outputs[1], seq_lengths = inputs_len, seq_axis = 1)
                outputs = (fw, bw)
                               
            return tf.concat(outputs, 2), tf.concat(states, axis=2)
            
                   
        
        else:
            outputs, states = tf.nn.dynamic_rnn(
                                                cell = cell_fw,
                                                inputs = inputs,
                                                sequence_length = inputs_len,
                                                dtype = tf.float32,
                                                scope = scope,
                                                time_major = False)
            return outputs, states