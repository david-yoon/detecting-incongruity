
######################################
# News Dataset (aaai-19)
# 
######################################
CUDA_VISIBLE_DEVICES=0 python train_AHDE.py --batch_size 256 --encoder_size 80 --context_size 10 --encoderR_size 49 --num_layer 1 --hidden_dim 300  --num_layer_con 1 --hidden_dim_con 300 --embed_size 300 --lr 0.001 --num_train_steps 100000 --is_save 0 --graph_prefix 'ahde' --corpus 'aaai-19_whole'


#########################################################
# NELA-18 dataset (avg, std, max, avg+2std, avg+3std)
# numpy (22, 2000)
# TITLE:         11.4,   3.4,    25,   18.2,   22.0
# BODY :        690.4, 644.4, 30917, 1987.4, 2631.8
# #CON  :        17.0,  17.8,  1684,   52.7,   70.6
# BODY in CON :  40.1,  46.7, 15395,  133.6,  180.4
#########################################################
CUDA_VISIBLE_DEVICES=0 python train_AHDE.py --batch_size 64 --encoder_size 180 --context_size 50 --encoderR_size 22 --num_layer 1 --hidden_dim 100  --num_layer_con 1 --hidden_dim_con 100 --embed_size 300 --use_glove 1 --lr 0.001 --num_train_steps 100000 --is_save 0 --graph_prefix 'ahde' --corpus 'nela-17_whole'

