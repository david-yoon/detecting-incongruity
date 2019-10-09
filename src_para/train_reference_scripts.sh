################################################################################
# News Dataset (aaai-19) (avg, std, max, avg+2std(95%), avg+3std(99.7%) )
# numpy (49, 170)
# TITLE:         13.7,   3.5,    48,   20.8,    24.4    --> encoderR_size
# BODY :         57.0,  37.3,   170,  131.7,   169.0
# CON  :          1.8,   1.3,    28,    4.5,     5.9    --> context_size
# BODY in CON :  29.1,  15.1,   168,   59.4,    74.5    --> encoder_size
################################################################################

CUDA_VISIBLE_DEVICES=0 python AHDE_Model.py --batch_size 256 --encoder_size 60 --context_size 8 --encoderR_size 49 --num_layer 1 --hidden_dim 300  --num_layer_con 1 --hidden_dim_con 300 --embed_size 300 --lr 0.001 --num_train_steps 100000 --is_save 1 --graph_prefix 'ahde_para'


CUDA_VISIBLE_DEVICES=0 python AHDE_Model.py --batch_size 256 --encoder_size 75 --context_size 6 --encoderR_size 25 --num_layer 1 --hidden_dim 300  --num_layer_con 1 --hidden_dim_con 300 --embed_size 300 --lr 0.001 --num_train_steps 100000 --is_save 1 --graph_prefix 'ahde_para'