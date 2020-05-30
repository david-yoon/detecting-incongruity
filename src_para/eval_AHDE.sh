# CUDA_VISIBLE_DEVICES=0 python eval_AHDE.py --model_path model_ahde --batch_size 2000 --encoder_size 60 --context_size 8 --encoderR_size 49 --num_layer 1 --hidden_dim 300  --num_layer_con 1 --hidden_dim_con 300 --embed_size 300 --is_test 1


CUDA_VISIBLE_DEVICES=1 python eval_AHDE.py --model_path model_ahde --batch_size 2000 --encoder_size 60 --context_size 8 --encoderR_size 49 --num_layer 1 --hidden_dim 300  --num_layer_con 1 --hidden_dim_con 300 --embed_size 300 --is_test 1
