#-*- coding: utf-8 -*-

"""
what    : AHDE evaluation
"""

from tensorflow.core.framework import summary_pb2
from random import shuffle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from util_pick_max import pick_max_prob
from tqdm import tqdm

"""
    desc  : similar to evaluation in whole-hrde, excpets para-method needs to consider the differences btw data & label
    
    inputs: 
        sess: tf session
        model: model for test
        data: such as the valid_set, test_set...        
    
    return:
        sum_batch_ce : sum( whole cross_entropy )
        accr   : accuracy
        probs  : 
        labels : 
"""
def run_test(sess, model, batch_gen, data, is_testset=False):
    
    batch_ce = []
    labels = []
    probs = []
    correct = []

    softmax, ce = 0, 0
    test_index = 0
    
    itr_loop = len(data) / model.batch_size
    
    # run 1 more time ( for batch remaining )
    for test_itr in tqdm(xrange( itr_loop + 1 )):
        
        raw_encoder_inputs, raw_encoderR_inputs, raw_encoder_seq, raw_context_seq, raw_encoderR_seq, raw_target_label = batch_gen.get_batch(
                                                                            data=data,
                                                                            batch_size=model.batch_size,
                                                                            encoder_size=model.encoder_size,
                                                                            context_size = model.context_size,
                                                                            encoderR_size=model.encoderR_size,
                                                                            is_test=True,
                                                                            start_index= (test_itr* model.batch_size)
                                                                            )

        # prepare data which will be push from pc to placeholder
        input_feed = {}

        input_feed[model.encoder_inputs] = raw_encoder_inputs
        input_feed[model.encoderR_inputs] = raw_encoderR_inputs

        input_feed[model.encoder_seq_length] = raw_encoder_seq
        input_feed[model.context_seq_length] = raw_context_seq
        input_feed[model.encoderR_seq_length] = raw_encoderR_seq

        input_feed[model.y_label] = raw_target_label

        # no drop out while evaluating
        input_feed[model.dr_text_in_ph]   = 1.0
        input_feed[model.dr_text_out_ph] = 1.0
        input_feed[model.dr_con_in_ph]   = 1.0
        input_feed[model.dr_con_out_ph] = 1.0
        
        input_feed[model.dr_memory_prob] = 1.0

        lo = None
        bprob = None
        
        try:
            bprob, b_loss, lo = sess.run([model.batch_prob, model.batch_loss, model.loss], input_feed)
        except:
            print "excepetion occurs in valid step : " + str(test_itr)
            pass
        
        batch_ce.append( lo )
        probs.extend( bprob )
        labels.extend( [x for x in raw_target_label ] )

        
    # cut-off dummy data used for batch
    batch_ce = batch_ce[:len(data)]
    probs    = probs[:len(data)]
    labels   = labels[:len(data)]
    ids      = batch_gen.test_data['id'][:len(data)]
        
    if (is_testset) :
        print 'test case'
        probs = pick_max_prob(ids, probs)
        labels   = labels[:len(probs)]

        
    pred_from_probs = [ 1 if x >= 0.5 else 0 for x in probs ]
    accr = accuracy_score(y_true=labels, y_pred=pred_from_probs)
    auroc = roc_auc_score(y_true=labels, y_score=probs)

    with open('./output.txt', 'wb') as f:
        for pr in probs:
            f.write(str(pr[0]) + '\n')
    
    sum_batch_ce = sum(batch_ce)
    
    value1 = summary_pb2.Summary.Value(tag="valid_loss", simple_value=sum_batch_ce)
    value2 = summary_pb2.Summary.Value(tag="valid_accuracy", simple_value=accr )
    value3 = summary_pb2.Summary.Value(tag="valid_auroc", simple_value=auroc )
    summary = summary_pb2.Summary(value=[value1, value2, value3])
    
    return sum_batch_ce, accr, probs, auroc, summary