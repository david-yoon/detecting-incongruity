#-*- coding: utf-8 -*-

'''
evaluate AHDE
'''
import csv
from AHDE_Model import *
from AHDE_process_data import *
from AHDE_evaluation import *

import os
import time
import argparse
from random import shuffle
from params import Params


def main(model_path, batch_size, encoder_size, context_size, encoderR_size, num_layer, hidden_dim, num_layer_con, hidden_dim_con,
         embed_size, num_train_steps, lr, valid_freq, is_save, is_test, 
         use_glove, fix_embed,
         memory_dim, topic_size):

    
    batch_gen = ProcessData(is_test=is_test)
    
    model = AttnHrDualEncoderModel(
                                voca_size=len(batch_gen.voca),
                                batch_size=batch_size,
                                encoder_size=encoder_size,
                                context_size = context_size,
                                encoderR_size=encoderR_size,
                                num_layer=num_layer,
                                hidden_dim=hidden_dim,
                                num_layer_con=num_layer_con,
                                hidden_dim_con=hidden_dim_con,
                                lr = lr,
                                embed_size=embed_size,
                                use_glove = use_glove,
                                fix_embed = fix_embed,
                                memory_dim=memory_dim,
                                topic_size=topic_size
                            )
        
    model.build_graph()
    
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        model_path = model_path + '/'
        ckpt = tf.train.get_checkpoint_state( os.path.dirname(model_path) )

        print 'model_path = ' + model_path
        
        if ckpt and ckpt.model_checkpoint_path:
            print ('from check point!!!')
            print ('from check point!!!')
            print ('from check point!!!')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else :
            print ('Can\'t load model... ERROR')
            print ('Can\'t load model... ERROR')
            print ('Can\'t load model... ERROR')
            return

        test_ce, test_accr, test_probs, test_auroc, _ = run_test(sess=sess, model=model, batch_gen=batch_gen,
                                                 data=batch_gen.test_set, is_testset=True)
            

        print   "data size: " + str(len(test_probs)) + '\t' + \
                "  test accr: " + '{:.3f}'.format(test_accr) + \
                "  test auroc: " + '{:.3f}'.format(test_auroc) + \
                "  loss: " + '{:.2f}'.format(test_ce)  

    
if __name__ == '__main__':
    
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_path', type=str, default="")    
    
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
    
    args = p.parse_args()
    
    main(
        model_path=args.model_path,
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
        is_test=args.is_test,
        use_glove=args.use_glove,
        fix_embed=args.fix_embed,
        memory_dim=args.memory_dim,
        topic_size=args.topic_size
        )