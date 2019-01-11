#-*- coding: utf-8 -*-

import numpy as np

"""
    desc  : 
    
    inputs: 
        ids  : list of ids
        probs: list of probs ( len(ids) == len(probs) )
    
    return:
        final_max_prob : max_prob for each ids
"""
def pick_max_prob(ids, probs):

    if len(ids) != len(probs) :
        print 'size shold be same among ids,probs: ', len(ids), len(probs)
        return None
    
    ids.append('LAST_DUMMY_ID')
    probs.append(np.asarray( [0.0], dtype=np.float32))
    
    final_max_prob = []
    
    for n, row in enumerate( zip(ids, probs) ):

        curr_id = row[0]
        curr_prob = row[1]
        if n == 0:
            max_prob = curr_prob
        else:
            if prev_id == curr_id:
                if max_prob < curr_prob:
                    max_prob = curr_prob
            else:
                final_max_prob.append(max_prob)
                max_prob = curr_prob

        prev_id = curr_id
        
    return final_max_prob
