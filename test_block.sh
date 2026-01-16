#!/bin/bash
####nohup   ./longjobs.sh > longjobs.out 2>&1 &
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/userhome/code/MinkLoc3D-master
. ~/.bashrc

###point cloud branch block tests
##这里我直接修改成了infall的结果
## ECA
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_ECA.txt >wavemulinferall_eca.log

## bottle
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_botttle.txt >wavemulinferall_bottle.log

##Wait to train
## ECA
#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/wavemultimodal_ECA.txt >wavemulinferall_RGBS_eca.log
## bottle
#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/wavemultimodal_botttle.txt >wavemulinferall_RGBS_bottle.log


## ECA
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_ECA.txt >wavemulinfer_eca.log

## bottle
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_botttle.txt >wavemulinfer_bottle.log

