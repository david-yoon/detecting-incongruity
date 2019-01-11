# detecting-incongruity


## This repository contains the source code & data corpus used in the following paper,

**Detecting Incongruity Between News Headline and Body Text via a Deep Hierarchical Encoder**, AAAI-19, <a href="https://arxiv.org/abs/1811.07066">paper</a>

----------

### [requirements]
	tensorflow==1.4 (tested on cuda-8.0, cudnn-6.0)
	python==2.7
	scikit-learn==0.20.0
	nltk==3.3


### [download data corpus]
- download preprocessed dataset with the following script
	> cd data <br>
	> sh download_dataset.sh
- downloaded dataset will be placed into following path of the project
	>	/data/para <br>
	>	/data/whole


### [source code]
- according to the training method
	 >	 whole-type: using the codes in the src_whole <br>
	 >	 para-type: using the codes in the src_para <br>

----------


### [training phase]
- each source code folder contains a training script
	> << for example >> <br>
	> /src_whole/ <br>
	> ./train_AHDE.sh : train dataset with AHDE model and "whole" method <br>
- results will be displayed in console <br>


<space>**※ hyper parameters**
- major parameters : edit from "train_AHDE.sh" <br>
- other parameters : edit from "/src_whole/params.py"

### [inference phase]
- each source code folder contains a inference script
- you need to modify the "model_path" in the "eval_AHDE.sh" to a proper path
	> << for example >> <br>
	> /src_whole/ <br>
	> ./eval_AHDE.sh   : evaluate test dataset with AHDE model and "whole" method
- results will be displayed in console <br>
- scores for the testset will be stored in "output.txt" <br>


----------


### [cite]
- Please cite our paper, when you use our code | dataset | model

  >	@article{yoon2018detecting, <br>
  >		title={Detecting Incongruity Between News Headline and Body Text via a Deep Hierarchical Encoder}, <br>
  >		author={Yoon, Seunghyun and Park, Kunwoo and Shin, Joongbo and Lim, Hongjun and Won, Seungpil and Cha, Meeyoung and Jung, Kyomin}, <br>
  >		journal={arXiv preprint arXiv:1811.07066}, <br>
  >		year={2018} <br>
  >		}