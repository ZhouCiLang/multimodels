#!/bin/bash
####nohup   ./longjobs.sh > longjobs.out 2>&1 &
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/userhome/code/MinkLoc3D-master
. ~/.bashrc

/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergb256.txt > ./logs/ablations/wavergb18_output256.log

####multi-modal
## ResNet34
#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinfer_rgb34.txt >./logs/ablations/wavemultimodalinferall256_rgb34.log

#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergbtest.txt >./logs/ablations/wavergb34_256.log


##only rgb
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergbtest.txt >./logs/ablations/wavergb_34_256.log

#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergb.txt> wavergb18_output128.log
#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergbtest.txt> wavergb_34_output128.log


##wait to run
## ResNet34
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/wavemultimodalinfer_rgb34.txt >./logs/ablations/wavemultimodalinferall256_rgbs_rgb34.log

# ##only rgb
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_rgb_sketch.txt --model_config ./models/wavergbtest.txt >./logs/ablations/wavergbsketch_34_256.log


###net architectures
#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergb.txt> wavergb18_output128.log
#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergbtest.txt> wavergb_34_output128.log
#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergbtest2.txt> wavergb_50_output128.log