
#-*- coding: utf-8 -*-

"""
what    : train Attentive Hierarchical Dual Encoder (AHDE) Model 
"""

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import DropoutWrapper 

from tensorflow.core.framework import summary_pb2

from AHDE_Model import AttnHrDualEncoderModel
from AHDE_process_data import *
from AHDE_evaluation import *
from params import *

import os
import time
import argparse
import datetime


        
# for training         
def train_step(sess, model, batch_gen):
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
        early_stop_count = params.MAX_EARLY_STOP_COUNT
        
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
                summary = train_step(sess, model, batch_gen)
                writer.add_summary( summary, global_step=model.global_step.eval() )
                
            except:
                print "excepetion occurs in train step"
                pass
                
            
            # run validation
            if (index + 1) % valid_freq == 0:
                
                dev_ce, dev_accr, dev_probs, dev_auroc, dev_summary = run_test(sess=sess, model=model, batch_gen=batch_gen,
                                             data=batch_gen.valid_set, IS_TEST=False)
                
                writer.add_summary( dev_summary, global_step=model.global_step.eval() )
                
                end_time = time.time()

                if index > params.CAL_ACCURACY_FROM:

                    if ( dev_ce < MIN_CE ):
                        MIN_CE = dev_ce

                        # save best result
                        if is_save is 1:
                            saver.save(sess, 'save/' + graph_dir_name + '/', model.global_step.eval() )

                        early_stop_count = params.MAX_EARLY_STOP_COUNT
                        
                        test_ce, test_accr, test_probs, test_auroc, test_summary = run_test(sess=sess, model=model, batch_gen=batch_gen,
                                            data=batch_gen.test_set, IS_TEST=True)
                        
                        writer.add_summary( test_summary, global_step=model.global_step.eval() )
                        
                        best_dev_accr = dev_accr
                        best_test_accr = test_accr
                        best_test_auroc = test_auroc
                        
                    else:
                        # early stopping
                        if early_stop_count == 0:
                            print "early stopped by no improvement: ", params.MAX_EARLY_STOP_COUNT
                            break
                             
                        test_accr = 0
                        early_stop_count = early_stop_count -1

                    print str( int(end_time - initial_time)/60 ) + " mins" + \
                        " step/seen/itr: " + str( model.global_step.eval() ) + "/ " + \
                                               str( model.global_step.eval() * model.batch_size ) + "/" + \
                                               str( round( model.global_step.eval() * model.batch_size / float(len(batch_gen.train_set)), 2)  ) + \
                        "\tval: " + '{:.3f}'.format(dev_accr)  +  \
                        "  tt: " + '{:.3f}'.format(test_accr) + \
                        "  troc: " + '{:.3f}'.format(test_auroc) + \
                        "  loss: " + '{:.2f}'.format(dev_ce)    
                
        writer.close()
            
        # result logging to file
        with open('./TEST_run_result.txt', 'a') as f:
            f.write(
                    #datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + '\t' + \
                    graph_dir_name + '\t' + \
                    str('d_accr:') + str(best_dev_accr) + '\t' + \
                    str('t_accr:') + str(best_test_accr) + '\t' + \
                    str('t_auroc:') + str(best_test_auroc) + '\t' + \
                    '\n') 


def create_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        
        
def main(params,
         batch_size, encoder_size, context_size, encoderR_size, num_layer, hidden_dim, num_layer_con, hidden_dim_con,
         embed_size, num_train_steps, lr, valid_freq, is_save, graph_dir_name, is_test, 
         use_glove, fix_embed
        ):
    
    if is_save is 1:
        create_dir('save/')
        create_dir('save/'+ graph_dir_name )
    
    create_dir('graph/')
    create_dir('graph/' + graph_dir_name )
    
    batch_gen = ProcessData(params, is_test=is_test)
    if is_test == 1:
        valid_freq = 100
    
    model = AttnHrDualEncoderModel(
                               params, 
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
                               fix_embed = fix_embed
                               )
    
    model.build_graph()
    
    valid_freq = int( len(batch_gen.train_set) * params.EPOCH_PER_VALID_FREQ / float(batch_size)  ) + 1
    train_model(params, model, batch_gen, num_train_steps, valid_freq, is_save, graph_dir_name)
    
if __name__ == '__main__':
    
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', type=str, default='aaai19_whole')
    
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
    
    args = p.parse_args()
    
    if args.corpus == ('aaai19_whole'): 
        params    = Params()
        params.DATA_DIR = '../data/target_aaai_whole/'
    
    
    elif args.corpus == ('nela18_whole'): 
        params    = Params_NELA()
        params.DATA_DIR = '../data/target_nela18_whole/'
    
    
    graph_name = args.graph_prefix + \
                    '_' + str(args.corpus) + \
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

    if params.dr_text_in   != 1.0 : graph_name = graph_name + '_drTi' +str(params.dr_text_in)
    if params.dr_text_out != 1.0 : graph_name = graph_name + '_drTo' +str(params.dr_text_out)
    if params.dr_con_in   != 1.0 : graph_name = graph_name + '_drCi' +str(params.dr_con_in)
    if params.dr_con_out != 1.0 : graph_name = graph_name + '_drCo' +str(params.dr_con_out)
                    
    if params.is_text_encoding_bidir  : graph_name = graph_name + '_Tbi'
    if params.is_chunk_encoding_bidir : graph_name = graph_name + '_Cbi'
    
    if params.is_text_residual  : graph_name = graph_name + '_TResi'
    if params.is_chunk_residual : graph_name = graph_name + '_CResi'
    
    if params.add_attention           : graph_name = graph_name + '_Attn'
    
    graph_name = graph_name + '_' + datetime.datetime.now().strftime("%m-%d-%H-%M")
    
    main(
        params=params,
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
        fix_embed=args.fix_embed
        )
