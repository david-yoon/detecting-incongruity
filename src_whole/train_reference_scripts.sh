
################################################################################
# News Dataset (aaai-19) (avg, std, max, avg+2std(95%), avg+3std(99.7%) )
# numpy (49, 1200)
# TITLE:         13.7,   3.5,    48,   20.8,    24.4    --> encoderR_size
# BODY :        499.8, 282.3,  1200, 1064.5,  1346.8
# CON  :          8.1,   5.0,   190,   18.2,    23.3    --> context_size
# BODY in CON :  59.6,  65.8,  1198,  191.3,   257.2    --> encoder_size
################################################################################
CUDA_VISIBLE_DEVICES=0 python AHDE_Model.py --batch_size 256 --encoder_size 80 --context_size 10 --encoderR_size 49 --num_layer 1 --hidden_dim 300  --num_layer_con 1 --hidden_dim_con 300 --embed_size 300 --lr 0.001 --num_train_steps 100000 --is_save 1 --graph_prefix 'ahde' --corpus 'aaai-19_whole' --data_path '../data/target_aaai-19_whole/'

CUDA_VISIBLE_DEVICES=0 python AHDE_Model.py --batch_size 128 --encoder_size 200 --context_size 23 --encoderR_size 49 --num_layer 1 --hidden_dim 300  --num_layer_con 1 --hidden_dim_con 300 --embed_size 300 --lr 0.001 --num_train_steps 100000 --is_save 1 --graph_prefix 'ahde' --corpus 'aaai-19_whole' --data_path '../data/target_aaai-19_whole/'



################################################################################
# NELA-17 dataset (avg, std, max, avg+2std(95%), avg+3std(99.7%) 
# numpy (25, 2000)
# TITLE:         12.3,   4.2,    79,   20.7,    25.0    --> encoderR_size
# BODY :        704.6, 641.8, 21113, 1988.3,  2630.2
# CON  :        13.5,  11.6,   367,   36.7,     48.3    --> context_size
# BODY in CON :  51.1,  50.5,  4709,  152.2,   202.7    --> encoder_size
################################################################################
CUDA_VISIBLE_DEVICES=0 python AHDE_Model.py --batch_size 64 --encoder_size 200 --context_size 50 --encoderR_size 25 --num_layer 1 --hidden_dim 100  --num_layer_con 1 --hidden_dim_con 100 --embed_size 300 --use_glove 1 --lr 0.001 --num_train_steps 100000 --is_save 1 --graph_prefix 'ahde' --corpus 'nela-17_whole' --data_path '../data/target_nela-17_whole/'
