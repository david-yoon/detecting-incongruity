# CUDA_VISIBLE_DEVICES=0 python eval_AHDE.py --model_path "model_ahde" --batch_size 250 --encoder_size 80 --context_size 10 --encoderR_size 49 --num_layer 1 --hidden_dim 300  --num_layer_con 1 --hidden_dim_con 300 --embed_size 300 --corpus 'aaai-19_whole' --data_path '../data/target_aaai-19_whole/'




CUDA_VISIBLE_DEVICES=1 python eval_AHDE.py --model_path "model_ahde-TEST" --batch_size 100 --encoder_size 170 --context_size 44 --encoderR_size 22 --num_layer 1 --hidden_dim 200  --num_layer_con 1 --hidden_dim_con 100 --embed_size 300 --corpus 'news-19_whole' --data_path '../data/headline_swap_news_v2.5/whole/'

