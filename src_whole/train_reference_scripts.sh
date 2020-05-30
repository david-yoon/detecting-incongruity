################################################################################
# Evaluation
################################################################################

CUDA_VISIBLE_DEVICES=1 python eval_AHDE.py --model_path "model_ahde-TEST" --batch_size 100 --encoder_size 170 --context_size 44 --encoderR_size 22 --num_layer 1 --hidden_dim 200  --num_layer_con 1 --hidden_dim_con 100 --embed_size 300 --corpus 'news-19_whole' --data_path '../data/headline_swap_news_v2.5/whole/'



################################################################################
# News Dataset (aaai-19) (avg, std, max, avg+2std(95%), avg+3std(99.7%) )
# numpy (49, 1200)
# TITLE:                  13.7,     3.5,      48,      20.8,       24.4    --> encoderR_size
# BODY :              499.8,  282.3,  1200,  1064.5,  1346.8
# CON  :                   8.1,      5.0,    190,      18.2,      23.3    --> context_size
# BODY in CON :    59.6,     65.8,  1198,    191.3,    257.2    --> encoder_size
################################################################################
CUDA_VISIBLE_DEVICES=0 python AHDE_Model.py --batch_size 256 --encoder_size 80 --context_size 10 --encoderR_size 49 --num_layer 1 --hidden_dim 300  --num_layer_con 1 --hidden_dim_con 300 --embed_size 300 --lr 0.001 --num_train_steps 100000 --is_save 1 --graph_prefix 'ahde' --corpus 'aaai-19_whole' --data_path '../data/target_aaai-19_whole/'

CUDA_VISIBLE_DEVICES=0 python AHDE_Model.py --batch_size 128 --encoder_size 200 --context_size 23 --encoderR_size 49 --num_layer 1 --hidden_dim 300  --num_layer_con 1 --hidden_dim_con 300 --embed_size 300 --lr 0.001 --num_train_steps 100000 --is_save 1 --graph_prefix 'ahde' --corpus 'aaai-19_whole' --data_path '../data/target_aaai-19_whole/'



################################################################################
# NELA-17 dataset (avg, std, max, avg+2std(95%), avg+3std(99.7%) 
# numpy (25, 2000)
# TITLE:                12.3,      4.2,         79,      20.7,      25.0    --> encoderR_size
# BODY :             704.6,  641.8,  21113,  1988.3,  2630.2
# CON  :                13.5,    11.6,      367,      36.7,      48.3    --> context_size
# BODY in CON :    51.1,    50.5,    4709,    152.2,    202.7    --> encoder_size
################################################################################
CUDA_VISIBLE_DEVICES=0 python AHDE_Model.py --batch_size 64 --encoder_size 200 --context_size 50 --encoderR_size 25 --num_layer 1 --hidden_dim 200  --num_layer_con 1 --hidden_dim_con 100 --embed_size 300 --use_glove 1 --lr 0.001 --num_train_steps 100000 --is_save 1 --graph_prefix 'ahde' --corpus 'nela-17_whole' --data_path '../data/target_nela-17_whole/'



################################################################################
# NELA-18 dataset (avg, std, max, avg+2std(95%), avg+3std(99.7%)   - To be released
# numpy (25, 2800)
# TITLE:               11.5,       3.4,        25,      18.3,      21.7    --> encoderR_size
# BODY :            657.1,  622.2,  30917,  1901.5,  2523.7
# CON        :        16.3,     17.9,    1679,      52.1,     70.0    --> context_size
# BODY in CON :  39.2,     43.3,    8613,    125.9,   169.3    --> encoder_size
################################################################################
CUDA_VISIBLE_DEVICES=0 python AHDE_Model.py --batch_size 40 --encoder_size 180 --context_size 70 --encoderR_size 25 --num_layer 1 --hidden_dim 200  --num_layer_con 1 --hidden_dim_con 100 --embed_size 300 --use_glove 1 --lr 0.001 --num_train_steps 100000 --is_save 1 --graph_prefix 'ahde' --corpus 'nela-18_whole' --data_path '../data/target_nela-18_whole/'



################################################################################
# NEWS-19 (head-swap) (avg, std, max, avg+2std(95%), avg+3std(99.7%)   - To be released
# numpy (25, 2100)
# TITLE:             11.0     3.9       65      18.8      22.7    --> encoderR_size
# BODY :            694.4   470.0    27362    1634.4    2104.5
# CON        :       15.4     8.3       51      32.2      40.6    --> context_size
# BODY in CON :      43.9    42.6    27074     129.2     171.8    --> encoder_size
################################################################################
CUDA_VISIBLE_DEVICES=0 python AHDE_Model.py --batch_size 40 --encoder_size 170 --context_size 40 --encoderR_size 22 --num_layer 1 --hidden_dim 200  --num_layer_con 1 --hidden_dim_con 100 --embed_size 300 --use_glove 1 --lr 0.001 --num_train_steps 100000 --is_save 1 --graph_prefix 'news-19-ahde' --corpus 'news-19_whole' --data_path '../data/target_news-19_headline/'


################################################################################
# NEWS-19 (para-swap) (avg, std, max, avg+2std(95%), avg+3std(99.7%)   - To be released
# numpy (25, 2100)
# TITLE:             11.0     3.8       57      18.8      22.7    --> encoderR_size
# BODY :            760.6   504.7    27362    1770.1    2104.9
# CON        :       16.7     9.2       96      35.2      44.4    --> context_size
# BODY in CON :      44.4    43.0    27074     130.5     173.6    --> encoder_size
################################################################################
CUDA_VISIBLE_DEVICES=0 python AHDE_Model.py --batch_size 40 --encoder_size 170 --context_size 44 --encoderR_size 22 --num_layer 1 --hidden_dim 200  --num_layer_con 1 --hidden_dim_con 100 --embed_size 300 --use_glove 1 --lr 0.001 --num_train_steps 100000 --is_save 1 --graph_prefix 'news-19-ahde' --corpus 'news-19_whole' --data_path '../data/target_news-19_paragraph/'




