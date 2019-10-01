CUDA_VISIBLE_DEVICES=0 python train_AHDE.py --batch_size 256 --encoder_size 80 --context_size 10 --encoderR_size 49 --num_layer 1 --hidden_dim 300  --num_layer_con 1 --hidden_dim_con 300 --embed_size 300 --lr=0.001 --num_train_steps 100000 --is_save 1 --graph_prefix 'ahde' --corpus 'aaai19_whole'


CUDA_VISIBLE_DEVICES=0 python AHDE_Model.py --batch_size 64 --encoder_size 200 --context_size 45 --encoderR_size 25 --num_layer 1 --hidden_dim 100  --num_layer_con 1 --hidden_dim_con 100 --embed_size 300 --use_glove 1 --lr 0.001 --num_train_steps 100000 --is_save 0 --graph_prefix 'ahde' --data_path '../data/target_NELA'  --corpus 'nela18_whole'

