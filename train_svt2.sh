#!/bin/bash
####nohup   ./longjobs.sh > longjobs.out 2>&1 &
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/userhome/code/MinkLoc3D-master
. ~/.bashrc

##test svt series:


# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtinfer128.txt >./logs/svtinfer128.txt

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtinfer256.txt >./logs/svtinfer256.txt


# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtqiinfer128.txt >./logs/svtqiinfer128.txt

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtqiinfer256.txt >./logs/svtqiinfer256.txt


# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal352.txt --model_config ./models/svtinfer256.txt >./logs/svt/all352.log


#### loss_weights
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal1000.txt --model_config ./models/svtinfer128.txt >./logs/svt/all1000.log

##4,3,3, average case
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal433.txt --model_config ./models/svtinfer128.txt >./logs/svt/all433.log

##5,0,5  image master
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal505.txt --model_config ./models/svtinfer128.txt >./logs/svt/all505.log

##5, 1, 4
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal514.txt --model_config ./models/svtinfer128.txt >./logs/svt/all514.log

## 5, 2,3
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal523.txt --model_config ./models/svtinfer128.txt >./logs/svt/all523.log

##5,5,0 point cloud master
## 5,4,1
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal541.txt --model_config ./models/svtinfer128.txt >./logs/svt/all541.log

## 5,3,2
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal532.txt --model_config ./models/svtinfer128.txt >./logs/svt/all532.log


