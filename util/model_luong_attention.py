#-*- coding: utf-8 -*-

import tensorflow as tf


'''
desc : apply luong attention to target vector with given condition
input :
   - batch_size             : 
   - target                 : [batch, seq, embed]
   - condition              : [batch, embed] --> last hidden
   - target_encoder_length  : max encoder length
   - hidden                 : should be same btw target and condition, otherwise code should be changed
output : 
   - attented target : weighted sum [batch, embed]
   - norm_dot : attention weight
'''
def luong_attention( batch_size, target, condition, target_encoder_length, hidden_dim ) :

    # same dim [batch, max_seq, embed]
    batch_seq_embed_target = tf.reshape( target, [batch_size, target_encoder_length, hidden_dim] )
    

    batch_embed_given = condition
    batch_seq_embed_given = tf.reshape( batch_embed_given, [batch_size,  hidden_dim, 1] )


    # calculate similarity 
    dot = tf.matmul( batch_seq_embed_target,  batch_seq_embed_given )
    
    
    # pad goes to -inf --> goes "0" after softmax
    pad_position = tf.equal(tf.reshape(dot, [batch_size, target_encoder_length]), 0.0)
    tmp = tf.to_float(pad_position) * -1e9
    tmp = tf.expand_dims(tmp, 2)
    base = tf.ones( [batch_size, target_encoder_length, 1] ) * tmp
    
    norm_dot = tf.nn.softmax( dot+base, dim=1 )
   
    # weighted sum by using similarity (normalized)
    target_mul_norm = tf.multiply( batch_seq_embed_target, norm_dot )
    weighted_sum = tf.reduce_sum( target_mul_norm, axis=1 )
    
    return weighted_sum, norm_dot


    
'''
desc : apply luong attention to target vector with given condition

input :
   - batch_size             : 
   - target                 : [batch, seq, embed]
   - condition              : [batch, embed] --> last hidden
   - target_encoder_length  : max encoder length
   - hidden                 : should be same btw target and condition, otherwise code should be changed

output : 
   - attented target : weighted sum [batch, embed]
   - norm_dot : attention weight
'''
def luong_attention_new( batch_size, target, condition, batch_seq, max_len, hidden_dim ) :

    # same dim [batch, max_seq, embed]
    batch_seq_embed_target = tf.reshape( target, [batch_size, max_len, hidden_dim] )
    

    batch_embed_given = condition
    batch_seq_embed_given = tf.reshape( batch_embed_given, [batch_size,  hidden_dim, 1] )

    # calculate similarity 
    dot = tf.matmul( batch_seq_embed_target,  batch_seq_embed_given )
    dot = tf.squeeze(dot)
    
    
    """
    # pad goes to -inf --> goes "0" after softmax
    """
    mask = tf.sequence_mask( lengths=batch_seq, maxlen=max_len, dtype=tf.float32 )
    mask_value = -tf.ones_like( mask ) * tf.float32.max
    mask_value = tf.multiply( mask_value, ( 1- mask ) )
    base = mask_value
    
    norm_dot = tf.nn.softmax( dot + base, dim=-1 )
   
    # weighted sum by using similarity (normalized)
    target_mul_norm = tf.multiply( batch_seq_embed_target, tf.expand_dims(norm_dot, -1) )
    weighted_sum = tf.reduce_sum( target_mul_norm, axis=1 )
    
    return weighted_sum, norm_dot