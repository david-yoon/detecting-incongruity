
#-*- coding: utf-8 -*-

"""
what    : Attentive Hierarchical Dual Encoder (AHDE) Model 
"""

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import DropoutWrapper 

from tensorflow.core.framework import summary_pb2

from AHDE_process_data import *
from AHDE_evaluation import *
from layers import add_GRU


class AttnHrDualEncoderModel:
    
    
    def __init__(self, 
                 params,
                 voca_size, batch_size,
                 encoder_size, context_size, encoderR_size, 
                 num_layer, hidden_dim,
                 num_layer_con, hidden_dim_con,
                 lr, embed_size, 
                 use_glove, fix_embed):
        
        self.params = params
        
        self.source_vocab_size = voca_size
        self.target_vocab_size = voca_size
        self.batch_size = batch_size
        self.encoder_size = encoder_size
        self.context_size = context_size
        self.encoderR_size = encoderR_size
        self.num_layers = num_layer
        self.hidden_dim = hidden_dim
        self.num_layers_con = num_layer_con
        self.hidden_dim_con = hidden_dim_con
        self.lr = lr
        self.embed_size = embed_size
        self.use_glove = use_glove
        self.fix_embed = fix_embed
        
        self.dr_text_in   = self.params.dr_text_in
        self.dr_text_out = self.params.dr_text_out
        self.dr_con_in   = self.params.dr_con_in
        self.dr_con_out = self.params.dr_con_out
        
        self.encoder_inputs = []
        self.context_inputs = []
        self.encoderR_inputs =[]
        self.y_label =[]

        self.M = None
        self.b = None
        
        self.y = None
        self.optimizer = None

        self.batch_loss = None
        self.loss = 0
        self.batch_prob = None
        
        # for global counter
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    """
    barch : batch size
    encoding_length: 
    time_step : 
    """
    def _create_placeholders(self):
        print '[launch] create placeholders'
        with tf.name_scope('data'):
            
            # [ batch X encoding_length, time_step (encoder_size) ]
            self.encoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name="encoder")
            # [ batch, time_step ] 
            self.encoderR_inputs = tf.placeholder(tf.int32, shape=[None, None], name="encoderR")
            
            
            # [ batch X encoding_length X time_step ]  
            self.encoder_seq_length = tf.placeholder(tf.int32, shape=[None], name="encoder_seq_len")
            # [ batch X encoding_length ]
            self.context_seq_length = tf.placeholder(tf.int32, shape=[None], name="context_seq_len")
            # [ batch X time_step ] 
            self.encoderR_seq_length = tf.placeholder(tf.int32, shape=[None], name="encoderR_seq_len")

            # [ batch, label ]
            self.y_label = tf.placeholder(tf.float32, shape=[None, None], name="label")
            
            self.dr_text_in_ph      = tf.placeholder(tf.float32, name="dropout_text_in")
            self.dr_text_out_ph    = tf.placeholder(tf.float32, name="dropout_text_out")
            self.dr_con_in_ph      = tf.placeholder(tf.float32, name="dropout_con_in")
            self.dr_con_out_ph    = tf.placeholder(tf.float32, name="dropout_con_out")

            self.dr_memory_prob = tf.placeholder(tf.float32, name="dropout_memory") # just for matching evaluation code with memory net version

            # for using pre-trained embedding
            self.embedding_placeholder = tf.placeholder(tf.float32, shape=[self.source_vocab_size, self.embed_size], name="embedding_placeholder")


    def _create_embedding(self):
        print '[launch] create embedding layer'
        with tf.name_scope('embed_layer'):
            
            if self.fix_embed==0 :
                IS_TRAIN = True
                print '[INFO] embedding train : TRUE'
            else :
                IS_TRAIN = False
                print '[INFO] embedding train : FALSE FALSE FALSE !!!'
            
            self.embed_matrix = tf.Variable(tf.random_normal([self.source_vocab_size, self.embed_size],
                                                             mean=0.0,
                                                             stddev=0.01,
                                                             dtype=tf.float32,                                                             
                                                             seed=None),
                                                             trainable = IS_TRAIN,
                                                             name='embed_matrix')
            
            self.embed_en       = tf.nn.embedding_lookup(self.embed_matrix, self.encoder_inputs, name='embed_encoder')
            self.embed_enR      = tf.nn.embedding_lookup(self.embed_matrix, self.encoderR_inputs, name='embed_encoderR')

            
    def _use_external_embedding(self):
        print '[launch] use pre-trained embedding'
        self.embedding_init = self.embed_matrix.assign(self.embedding_placeholder)

        
    def _create_gru_hrde_model(self):
        print '[launch] create encoding layer textBi/chunkBi, textResi/chunkResi', self.params.is_text_encoding_bidir, self.params.is_chunk_encoding_bidir, self.params.is_text_residual, self.params.is_chunk_residual
        
        with tf.name_scope('text_encoding_RNN') as scope:
            
            # match embedding_dim - rnn_dim to use residual connection            
            if self.params.is_text_residual:
                self.text_residual_matrix = tf.Variable(tf.random_normal([self.embed_size, self.hidden_dim],
                                                             mean=0.0,
                                                             stddev=0.01,
                                                             dtype=tf.float32,                                                             
                                                             seed=None),
                                                             trainable = True,
                                                             name='text_residual_projection')
                
                h = tf.matmul( tf.reshape(self.embed_en, [-1, self.embed_size]), self.text_residual_matrix )
                self.embed_en = tf.reshape( h, [self.batch_size * self.context_size, self.encoder_size, self.hidden_dim] )
                                          
                h_R = tf.matmul( tf.reshape(self.embed_enR, [-1, self.embed_size]), self.text_residual_matrix )
                self.embed_enR = tf.reshape( h_R, [self.batch_size, self.encoderR_size, self.hidden_dim] )
            
            
            # enoder RNN
            self.outputs_en, self.states_en = add_GRU(
                                                inputs= self.embed_en,
                                                inputs_len=self.encoder_seq_length,
                                                hidden_dim = self.hidden_dim,
                                                layers = self.num_layers,
                                                scope = 'text_encoding_RNN',
                                                reuse = False,
                                                dr_input_keep_prob  = self.dr_text_in_ph,
                                                dr_output_keep_prob = self.dr_text_out_ph,
                                                is_bidir = self.params.is_text_encoding_bidir,
                                                is_residual = self.params.is_text_residual
                                                )
            
            # response RNN
            self.outputs_enR, self.states_enR = add_GRU(
                                                inputs= self.embed_enR,
                                                inputs_len=self.encoderR_seq_length,
                                                hidden_dim = self.hidden_dim,
                                                layers = self.num_layers,
                                                scope = 'text_encoding_RNN',
                                                reuse = True,
                                                dr_input_keep_prob  = self.dr_text_in_ph,
                                                dr_output_keep_prob = self.dr_text_out_ph,
                                                is_bidir = self.params.is_text_encoding_bidir,
                                                is_residual = self.params.is_text_residual
                                                )
            
            self.final_encoder   = self.states_en[-1]
            self.final_encoderR  = self.states_enR[-1]
            
            self.final_encoder_dimension   = self.hidden_dim 
            self.final_encoderR_dimension  = self.hidden_dim
        
        
        with tf.name_scope('chunk_encoding_RNN') as scope:
            
            # make data for context input
            self.con_input = tf.reshape( self.final_encoder, [self.batch_size, self.context_size, self.final_encoder_dimension])
            
            # make data for context input
            self.con_inputR = tf.reshape( self.final_encoderR, [self.batch_size, 1, self.final_encoderR_dimension])
            
            
            # match rnn_dim - context_rnn_dim to use residual connection            
            if self.params.is_chunk_residual:
                self.chunk_residual_matrix = tf.Variable(tf.random_normal([self.final_encoder_dimension, self.hidden_dim_con],
                                                             mean=0.0,
                                                             stddev=0.01,
                                                             dtype=tf.float32,                                                             
                                                             seed=None),
                                                             trainable = True,
                                                             name='chunk_residual_projection')
                
                h = tf.matmul( tf.reshape(self.con_input, [-1, self.final_encoder_dimension]), self.chunk_residual_matrix )
                self.con_input = tf.reshape( h, [self.batch_size, self.context_size, self.hidden_dim_con] )
                                          
                h_R = tf.matmul( tf.reshape(self.con_inputR, [-1, self.final_encoderR_dimension]), self.chunk_residual_matrix )
                self.con_inputR = tf.reshape( h_R, [self.batch_size, 1, self.hidden_dim_con] )
                
            
            self.outputs_con, self.last_states_con = add_GRU(
                                                            inputs= self.con_input,
                                                            inputs_len=self.context_seq_length,
                                                            hidden_dim = self.hidden_dim_con,
                                                            layers = self.num_layers_con,
                                                            scope = 'chunk_encoding_RNN',
                                                            reuse = False,
                                                            dr_input_keep_prob  = self.dr_con_in_ph,
                                                            dr_output_keep_prob = self.dr_con_out_ph,
                                                            is_bidir = self.params.is_chunk_encoding_bidir,
                                                            is_residual = self.params.is_chunk_residual
                                                            )
        
            
            self.outputs_conR, self.last_states_conR = add_GRU(
                                                            inputs= self.con_inputR,
                                                            inputs_len= np.ones(self.batch_size, dtype=np.int).tolist(),
                                                            hidden_dim = self.hidden_dim_con,
                                                            layers = self.num_layers_con,
                                                            scope = 'chunk_encoding_RNN',
                                                            reuse = True,
                                                            dr_input_keep_prob  = self.dr_con_in_ph,
                                                            dr_output_keep_prob = self.dr_con_out_ph,
                                                            is_bidir = self.params.is_chunk_encoding_bidir,
                                                            is_residual = self.params.is_chunk_residual
                                                            )
            
            self.final_encoder   = self.last_states_con[-1]
            self.final_encoderR  = self.last_states_conR[-1]
        
        
            if self.params.is_chunk_encoding_bidir: 
                self.final_encoder_dimension   = self.hidden_dim_con * 2
                self.final_encoderR_dimension  = self.hidden_dim_con * 2
            else:
                self.final_encoder_dimension   = self.hidden_dim_con
                self.final_encoderR_dimension  = self.hidden_dim_con
        

    def _create_attention_layers(self):
        print '[launch] create attention layer'
        from model_luong_attention import luong_attention
        with tf.name_scope('attention_layer') as scope:
        
            self.final_encoder, self.attn_norm = luong_attention( batch_size = self.batch_size,
                                                             target = self.outputs_con,
                                                             condition = self.final_encoderR,
                                                             target_encoder_length = self.context_size,
                                                             hidden_dim = self.final_encoder_dimension
                                                            )
        
        
    def _create_output_layers(self):
        print '[launch] create output projection layer'        
           
        with tf.name_scope('output_layer') as scope:
            
            self.M = tf.Variable(tf.random_uniform([self.final_encoder_dimension, self.final_encoderR_dimension],
                                                   minval= -0.25,
                                                   maxval= 0.25,
                                                   dtype=tf.float32,
                                                   seed=None),
                                                   name="similarity_matrix")
            
            self.b = tf.Variable(tf.zeros([1], dtype=tf.float32), name="output_bias")
            
            
            # (c * M) * r + b
            tmp_y = tf.matmul(self.final_encoder, self.M)
            batch_y_hat = tf.reduce_sum( tf.multiply(tmp_y, self.final_encoderR), 1, keep_dims=True ) + self.b
            self.batch_prob = tf.sigmoid( batch_y_hat )
                
        
        with tf.name_scope('loss') as scope:
            self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=batch_y_hat, labels=self.y_label )
            self.loss = tf.reduce_mean( self.batch_loss  )
    
    
    def _create_optimizer(self):
        print '[launch] create optimizer'
        
        with tf.name_scope('optimizer') as scope:
            opt_func = tf.train.AdamOptimizer(learning_rate=self.lr)
            gvs = opt_func.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_norm(t=grad, clip_norm=1), var) for grad, var in gvs]
            self.optimizer = opt_func.apply_gradients(grads_and_vars=capped_gvs, global_step=self.global_step)
    
    
    def _create_summary(self):
        print '[launch] create summary'
        
        with tf.name_scope('summary'):
            tf.summary.scalar('mean_loss', self.loss)
            self.summary_op = tf.summary.merge_all()
    
    
    def build_graph(self):
        self._create_placeholders()
        self._create_embedding()
        
        if self.use_glove == 1:
            self._use_external_embedding()
            
        self._create_gru_hrde_model()
        if self.params.add_attention : self._create_attention_layers()
        
        self._create_output_layers()
        self._create_optimizer()
        self._create_summary()