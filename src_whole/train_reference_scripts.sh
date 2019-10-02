
######################################
# News Dataset (aaai-19)
# 
######################################
CUDA_VISIBLE_DEVICES=0 python train_AHDE.py --batch_size 256 --encoder_size 80 --context_size 10 --encoderR_size 49 --num_layer 1 --hidden_dim 300  --num_layer_con 1 --hidden_dim_con 300 --embed_size 300 --lr 0.001 --num_train_steps 100000 --is_save 0 --graph_prefix 'ahde' --corpus 'aaai-19_whole'


#########################################################
# NELA-18 dataset (avg, std, max, avg+2std, avg+3std)
# numpy (22, 2000)
# TITLE:         12.3,   4.2,    79,   20.7,   25.0
# BODY :        704.6, 641.8, 21113, 1988.3, 2630.2
# #CON  :        13.5,  11.6,   367,   36.7,   48.3
# BODY in CON :  51.1,  50.5,  4709,  152.2,  202.7
#########################################################
CUDA_VISIBLE_DEVICES=0 python train_AHDE.py --batch_size 64 --encoder_size 180 --context_size 50 --encoderR_size 22 --num_layer 1 --hidden_dim 100  --num_layer_con 1 --hidden_dim_con 100 --embed_size 300 --use_glove 1 --lr 0.001 --num_train_steps 100000 --is_save 0 --graph_prefix 'ahde' --corpus 'nela-17_whole'

