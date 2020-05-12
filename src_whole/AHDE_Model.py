
#-*- coding: utf-8 -*-

"""
what    : Attentive Hierarchical Dual Encoder (AHDE) Model 
"""

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import DropoutWrapper 

from tensorflow.core.framework import summary_pb2
from random import shuffle

from AHDE_process_data import *
from AHDE_evaluation import *
from layers import add_GRU

import os
import sys
import time
import argparse
import datetime
from random import shuffle

class AttnHrDualEncoderModel:
    
    
    def __init__(self, params, voca_size, batch_size,
                 encoder_size, context_size, encoderR_size, 
                 num_layer, hidden_dim,
                 num_layer_con, hidden_dim_con,
                 lr, embed_size, 
                 use_glove, fix_embed,
                 memory_dim=0, topic_size=0):
        
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
        self.dr_con_out  = self.params.dr_con_out
        
        self.memory_dim = memory_dim
        self.topic_size = topic_size
    
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

            self.lr_ph            = tf.placeholder(tf.float32, name="dropout_con_out")
            
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
            opt_func = tf.train.AdamOptimizer(learning_rate=self.lr_ph)
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

        
# for training         
def train_step(sess, params, model, batch_gen, index):
    raw_encoder_inputs, raw_encoderR_inputs, raw_encoder_seq, raw_context_seq, raw_encoderR_seq, raw_target_label = batch_gen.get_batch(
                                        data=batch_gen.train_set,
                                        batch_size=model.batch_size,
                                        encoder_size=model.encoder_size,
                                        context_size=model.context_size,
                                        encoderR_size=model.encoderR_size,
                                        is_test=False
                                        )

    # prepare data which will be push from pc to placeholder
    input_feed = {}
    
    input_feed[model.encoder_inputs] = raw_encoder_inputs
    input_feed[model.encoderR_inputs] = raw_encoderR_inputs

    input_feed[model.encoder_seq_length] = raw_encoder_seq
    input_feed[model.context_seq_length] = raw_context_seq
    input_feed[model.encoderR_seq_length] = raw_encoderR_seq

    input_feed[model.y_label] = raw_target_label
    
    input_feed[model.dr_text_in_ph] = model.dr_text_in
    input_feed[model.dr_text_out_ph] = model.dr_text_out
    
    input_feed[model.dr_con_in_ph] = model.dr_con_in
    input_feed[model.dr_con_out_ph] = model.dr_con_out
    
    if params.APPLY_LR_DECAY:
        run_per_epoch = len(batch_gen.train_set) / float(model.batch_size) * params.DECAY_FREQ
        num_decay = int(index / run_per_epoch)
        input_feed[model.lr_ph] = model.lr * pow(params.DECAY_RATE, num_decay)
#         print('lr decay: index, num_per_epoch, num_decay, lr', index, run_per_epoch, num_decay, model.lr * pow(params.DECAY_RATE, num_decay))
    else:
        input_feed[model.lr_ph] = model.lr
    
    _, summary = sess.run([model.optimizer, model.summary_op], input_feed)
    
    return summary

    
def train_model(params, model, batch_gen, num_train_steps, valid_freq, is_save=0, graph_dir_name='default'):
    
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    summary = None
    dev_summary = None
    
    with tf.Session(config=config) as sess:
        
        sess.run(tf.global_variables_initializer())
        early_stop_count = model.params.MAX_EARLY_STOP_COUNT
        
        if model.use_glove == 1:
            sess.run(model.embedding_init, feed_dict={ model.embedding_placeholder: batch_gen.get_glove() })
            print 'use pre-trained embedding vector'

        
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('save/' + graph_dir_name + '/'))
        if ckpt and ckpt.model_checkpoint_path:
            print ('from check point!!!')
            saver.restore(sess, ckpt.model_checkpoint_path)
            

        writer = tf.summary.FileWriter('./graph/'+graph_dir_name, sess.graph)

        initial_time = time.time()
        
        MIN_CE = 1000000
        dev_accr = 0
        test_accr = 0
        
        best_dev_accr = 0
        best_test_accr = 0
        best_test_auroc = 0
        
        for index in xrange(num_train_steps):

            try:
                # run train 
                summary = train_step(sess, params, model, batch_gen, index)
                writer.add_summary( summary, global_step=model.global_step.eval() )
                
            except Exception as ex:
                print "excepetion occurs in train step"
                print ex
                pass
                
            
            # run validation
            if (index + 1) % valid_freq == 0:
                
                dev_ce, dev_accr, dev_probs, dev_auroc, dev_summary = run_test(sess=sess, model=model, batch_gen=batch_gen,
                                             data=batch_gen.valid_set)
                
                writer.add_summary( dev_summary, global_step=model.global_step.eval() )
                
                end_time = time.time()

                if index > model.params.CAL_ACCURACY_FROM:

                    if ( dev_ce < MIN_CE ):
                        MIN_CE = dev_ce

                        # save best result
                        if is_save is 1:
                            saver.save(sess, 'save/' + graph_dir_name + '/', model.global_step.eval() )

                        early_stop_count = model.params.MAX_EARLY_STOP_COUNT
                        
                        test_ce, test_accr, test_probs, test_auroc, _ = run_test(sess=sess, model=model, batch_gen=batch_gen,
                                            data=batch_gen.test_set)
                        
                        best_dev_accr = dev_accr
                        best_test_accr = test_accr
                        best_test_auroc = test_auroc
                        
                    else:
                        # early stopping
                        if early_stop_count == 0:
                            print "early stopped by no improvement: ", model.params.MAX_EARLY_STOP_COUNT
                            break
                             
                        test_accr = 0
                        early_stop_count = early_stop_count -1

                    print str( int(end_time - initial_time)/60 ) + " mins" + \
                        " step/seen/epoch: " + str( model.global_step.eval() ) + "/ " + \
                                               str( model.global_step.eval() * model.batch_size ) + "/" + \
                                               str( round( model.global_step.eval() * model.batch_size / float(len(batch_gen.train_set)), 2)  ) + \
                        "\tval: " + '{:.3f}'.format(dev_accr)  +  \
                        "  test: " + '{:.3f}'.format(test_accr) + \
                        "  troc: " + '{:.3f}'.format(test_auroc) + \
                        "  loss: " + '{:.2f}'.format(dev_ce)    
                
        writer.close()

        
        if (model.params.LAST_EVAL_TRAINSET):
        
            train_ce, train_accr, _, _, _ = run_test(sess=sess, model=model, batch_gen=batch_gen,
                                                                     data=batch_gen.train_set)

            print str( int(end_time - initial_time)/60 ) + " mins" + \
                            " step/seen/epoch: " + str( model.global_step.eval() ) + "/ " + \
                                                   str( model.global_step.eval() * model.batch_size ) + "/" + \
                                                   str( round( model.global_step.eval() * model.batch_size / float(len(batch_gen.train_set)), 2)  ) + \
                            "\tval: " + '{:.3f}'.format(dev_accr)  +  \
                            "  test: " + '{:.3f}'.format(test_accr) + \
                            "  troc: " + '{:.3f}'.format(test_auroc) + \
                            "  train_accr: " + '{:.3f}'.format(train_accr)    

        
        # result logging to file
        with open('./TEST_run_result.txt', 'a') as f:
            f.write(
                     " step/seen/epoch: " + str( model.global_step.eval() ) + "/ " + \
                       str( model.global_step.eval() * model.batch_size ) + "/" + \
                       str( round( model.global_step.eval() * model.batch_size / float(len(batch_gen.train_set)), 2)  ) + '\n' + \
                    #datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + '\t' + \
                    graph_dir_name + '\t' + \
                    str('d_accr:\t') + str(best_dev_accr) + '\t' + \
                    str('t_accr:\t') + str(best_test_accr) + '\t' + \
                    str('t_auroc:\t') + str(best_test_auroc) + '\t' + \
                    '\n') 


def create_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        
        
def main(params, data_path, batch_size, encoder_size, context_size, encoderR_size, num_layer, hidden_dim, num_layer_con, hidden_dim_con,
         embed_size, num_train_steps, lr, valid_freq, is_save, graph_dir_name, is_test, 
         use_glove, fix_embed,
         memory_dim, topic_size):
    
    if is_save is 1:
        create_dir('save/')
        create_dir('save/'+ graph_dir_name )
    
    create_dir('graph/')
    create_dir('graph/' + graph_dir_name )
    
    batch_gen = ProcessData(is_test=is_test, params=params, data_path=data_path)
    if is_test == 1:
        valid_freq = 100
    
    model = AttnHrDualEncoderModel(
                               params=params, 
                               voca_size=len(batch_gen.voca),
                               batch_size=batch_size,
                               encoder_size=encoder_size,
                               context_size=context_size,
                               encoderR_size=encoderR_size,
                               num_layer=num_layer,                 
                               hidden_dim=hidden_dim,
                               num_layer_con=num_layer_con,
                               hidden_dim_con=hidden_dim_con,
                               lr=lr,
                               embed_size=embed_size,
                               use_glove = use_glove,
                               fix_embed = fix_embed,
                               memory_dim = memory_dim,
                               topic_size=topic_size
                               )
    
    model.build_graph()
    
    valid_freq = int( len(batch_gen.train_set) * params.EPOCH_PER_VALID_FREQ / float(batch_size)  ) + 1
    train_model(params, model, batch_gen, num_train_steps, valid_freq, is_save, graph_dir_name)
    
if __name__ == '__main__':
    
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, default='../data/')
    
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--encoder_size', type=int, default=80)
    p.add_argument('--context_size', type=int, default=10)
    p.add_argument('--encoderR_size', type=int, default=80)
    
    # siaseme RNN
    p.add_argument('--num_layer', type=int, default=2)
    p.add_argument('--hidden_dim', type=int, default=300)
    
    # context RNN
    p.add_argument('--num_layer_con', type=int, default=2)
    p.add_argument('--hidden_dim_con', type=int, default=300)
    
    p.add_argument('--embed_size', type=int, default=200)
    p.add_argument('--num_train_steps', type=int, default=10000)
    p.add_argument('--lr', type=float, default=1e-1)
    p.add_argument('--valid_freq', type=int, default=500)
    p.add_argument('--is_save', type=int, default=0)
    p.add_argument('--graph_prefix', type=str, default="default")
    
    p.add_argument('--is_test', type=int, default=0)
    p.add_argument('--use_glove', type=int, default=0)
    p.add_argument('--fix_embed', type=int, default=0)
    
    # latent topic
    p.add_argument('--memory_dim', type=int, default=32)
    p.add_argument('--topic_size', type=int, default=0)
    
    p.add_argument('--corpus', type=str, default='')
    
    args = p.parse_args()
    
    graph_name = ''
                    
    if args.corpus == ('aaai-19_whole'):
        from params import Params
        print 'aaai-19'
        _params    = Params()
        graph_name = 'aaai-19'
        
    elif args.corpus == ('nela-17_whole'):
        from params import Params_NELA_17
        print 'nela-17'
        _params    = Params_NELA_17()
        graph_name = 'nela-17'
    
    elif args.corpus == ('nela-18_whole'):
        from params import Params_NELA_18
        print 'nela-18'
        _params    = Params_NELA_18()
        graph_name = 'nela-18'
      
    elif args.corpus == ('news-19_whole'):
        from params import Params_NEWS_19
        print 'news-19'
        _params    = Params_NEWS_19()
        graph_name = 'news-19'
    
    else:
        print('[ERROR] a corpus should be specified')
        sys.exit()    
            
    graph_name = graph_name + '_' + \
                args.graph_prefix + \
                '_b' + str(args.batch_size) + \
                '_es' + str(args.encoder_size) + \
                '_eRs' + str(args.encoderR_size) + \
                '_cs' + str(args.context_size) + \
                '_L' + str(args.num_layer) + \
                '_H' + str(args.hidden_dim) + \
                '_Lc' + str(args.num_layer_con) + \
                '_Hc' + str(args.hidden_dim_con) + \
                '_G' + str(args.use_glove) + \
                '_FIX' + str(args.fix_embed)
                
    
    if _params.add_LTC:
        graph_name = graph_name + \
                    '_M' + str(args.memory_dim) + \
                    '_T' + str(args.topic_size)

        
    if _params.dr_text_in   != 1.0 : graph_name = graph_name + '_drTi' +str(_params.dr_text_in)
    if _params.dr_text_out != 1.0 : graph_name = graph_name + '_drTo' +str(_params.dr_text_out)
    if _params.dr_con_in   != 1.0 : graph_name = graph_name + '_drCi' +str(_params.dr_con_in)
    if _params.dr_con_out != 1.0 : graph_name = graph_name + '_drCo' +str(_params.dr_con_out)
                    
    if _params.is_text_encoding_bidir  : graph_name = graph_name + '_Tbi'
    if _params.is_chunk_encoding_bidir : graph_name = graph_name + '_Cbi'
    
    if _params.is_text_residual  : graph_name = graph_name + '_TResi'
    if _params.is_chunk_residual : graph_name = graph_name + '_CResi'
    
    if _params.add_attention           : graph_name = graph_name + '_Attn'
        
    if _params.APPLY_LR_DECAY: graph_name = graph_name + '_lrDe'
    
    graph_name = graph_name + '_' + datetime.datetime.now().strftime("%m-%d-%H-%M")
    
    
    print'[INFO] data:\t\t', args.data_path
    print'[INFO] params:\t\t', args.corpus
    
    print'[INFO] batch:\t\t', args.batch_size
    
    print'[INFO] encoder_size:\t', args.encoder_size
    print'[INFO] context_size:\t', args.context_size
    print'[INFO] encoderR_size:\t', args.encoderR_size
    
    print'[INFO] num_layer:\t', args.num_layer
    print'[INFO] hidden_dim:\t', args.hidden_dim
    
    print'[INFO] num_layer_con:\t', args.num_layer_con
    print'[INFO] hidden_dim_con:\t', args.hidden_dim_con
    print'[INFO] embed_size:\t', args.embed_size
    
    print'[INFO] dr_text_in:\t', _params.dr_text_in
    print'[INFO] dr_text_out:\t', _params.dr_text_out
    print'[INFO] dr_con_in:\t', _params.dr_con_in
    print'[INFO] dr_con_out:\t', _params.dr_con_out
    
    
    print'[INFO] reverse_bw:\t', _params.reverse_bw
    print'[INFO] is_text_encoding_bidir:\t', _params.is_text_encoding_bidir
    print'[INFO] is_chunk_encoding_bidir:\t', _params.is_chunk_encoding_bidir
    print'[INFO] is_text_residual:\t', _params.is_text_residual
    print'[INFO] is_chunk_residual:\t', _params.is_chunk_residual
    
    print'[INFO] add_attention:\t', _params.add_attention
    print'[INFO] add_LTC:\t\t', _params.add_LTC
    
    print'[INFO] lr:\t\t', args.lr
    
    if _params.APPLY_LR_DECAY:
        print'[INFO] lr_decay_freq:\t', _params.DECAY_FREQ
        print'[INFO] lr_decay_rate:\t', _params.DECAY_RATE
    
    print'[INFO] valid_freq:\t', args.valid_freq
    print'[INFO] is_save:\t\t', args.is_save

    print'[INFO] is_test:\t\t', args.is_test
    print'[INFO] use_glove:\t', args.use_glove
    print'[INFO] fix_embed:\t', args.fix_embed
    
    main(
        params=_params,
        data_path=args.data_path,
        batch_size=args.batch_size,
        encoder_size=args.encoder_size,
        context_size=args.context_size,
        encoderR_size=args.encoderR_size,
        num_layer=args.num_layer,
        hidden_dim=args.hidden_dim,
        num_layer_con=args.num_layer_con,
        hidden_dim_con=args.hidden_dim_con,
        embed_size=args.embed_size, 
        num_train_steps=args.num_train_steps,
        lr=args.lr,
        valid_freq=args.valid_freq,
        is_save=args.is_save,
        graph_dir_name=graph_name,
        is_test=args.is_test,
        use_glove=args.use_glove,
        fix_embed=args.fix_embed,
        memory_dim=args.memory_dim,
        topic_size=args.topic_size
        )
