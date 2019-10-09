detecting-incongruity
------------------------------------------------------------

This repository contains the source code & data corpus used in the following paper,

**Detecting Incongruity Between News Headline and Body Text via a Deep Hierarchical Encoder**, AAAI-19, <a href="https://arxiv.org/abs/1811.07066">paper</a>

<img src="./assets/AHDE.png" width="50%">


Requirements
-------------
```
  tensorflow==1.4 (tested on cuda-8.0, cudnn-6.0)
  python==2.7
  scikit-learn==0.20.0
  nltk==3.3
```

Download Dataset
-------------
- download preprocessed dataset with the following script
	> cd data <br>
	> sh download_processed_dataset_aaai-19.sh
	
- the downloaded dataset will be placed into the following path of the project
	>	/data/aaai-19/para <br>
	>	/data/aaai-19/whole	

- format (example)
	> test_title.npy: [100000, 49] - (#samples, #token (index)) <br>
	> test_body: [100000, 1200] - (#samples, #token (index)) <br>
	> test_label: [100000] - (#samples) <br>
    > dic_mincutN.txt: dictionary

Source Code
-------------
- according to the training method
	 >	 whole-type: using the codes in the ./src_whole <br>
	 >	 para-type: using the codes in the ./src_para <br>



Training Phase
-------------
- each source code folder contains a reference script for training the model
	> train_reference_scripts.sh <br>
	> << for example >> <br>
	> train dataset with AHDE model and "whole" method <br>
```bash
python AHDE_Model.py --batch_size 256 --encoder_size 80 --context_size 10 --encoderR_size 49 --num_layer 1 --hidden_dim 300  --num_layer_con 1 --hidden_dim_con 300 --embed_size 300 --lr 0.001 --num_train_steps 100000 --is_save 1 --graph_prefix 'ahde' --corpus 'aaai-19_whole' --data_path '../data/target_aaai-19_whole/'
```
- Results will be displayed in the console <br>
- The final test result will be stored in "./TEST_run_result.txt" <br>


<space>**※ hyper parameters**
- major parameters: edit from the training script <br>
- other parameters: edit from "./params.py"


Inference Phase
-------------
- each source code folder contains an inference script
- you need to modify the "model_path" in the "eval_AHDE.sh" to a proper path
	> << for example >> <br>
	> evaluate test dataset with AHDE model and "whole" method <br>
```bash
	src_whole$ sh eval_AHDE.sh
```
- Results will be displayed in the console <br>
- scores for the testset will be stored in "./output.txt" <br>



Dataset Statistics
-------------
* whole case <br>

  | data  |  Samples  | tokens (avg)<br> headline| tokens (avg) <br> body text |
  |:-----:|:---------:|:------------:|:---------:|
  | train | 1,700,000 |        13.71 |    499.81 |
  |  dev  |   100,000 |        13.69 |    499.03 |
  |  test |   100,000 |        13.55 |    769.23 |

* Note <br>
	> We crawled articles for "dev" and "test" dataset from different media outlets. <br>


Newly introduced dataset (English version)
-------------
* We create an English version of the dataset, nela-17, using <a href="https://github.com/BenjaminDHorne/NELA2017-Dataset-v1">NELA 2017</a> data. Please refer to the dataset repository [<a href="https://github.com/sugoiii/detecting-incongruity-dataset-gen">link</a>].
* If you want to run our model (AHDE) with the nela-17 data, you can use the preprocessed dataset that is compatible with our code.
	> cd data <br>
	> sh download_processed_dataset_nela-17.sh
* training script (refer to the "train_reference_scripts.sh")
```bash
python AHDE_Model.py --batch_size 64 --encoder_size 200 --context_size 50 --encoderR_size 25 --num_layer 1 --hidden_dim 100  --num_layer_con 1 --hidden_dim_con 100 --embed_size 300 --use_glove 1 --lr 0.001 --num_train_steps 100000 --is_save 1 --graph_prefix 'ahde' --corpus 'nela-17_whole' --data_path '../data/target_nela-17_whole/'
```



cite
-------------
- Please cite our paper, when you use our code | dataset | model

	> @inproceedings{yoon2019detecting,<br>
> title={Detecting Incongruity between News Headline and Body Text via a Deep Hierarchical Encoder},<br>
> author={Yoon, Seunghyun and Park, Kunwoo and Shin, Joongbo and Lim, Hongjun and Won, Seungpil and Cha, Meeyoung and Jung, Kyomin},<br>
>  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},<br>
>  volume={33},<br>
>  pages={791--800},<br>
>  year={2019}<br>
> }
	
