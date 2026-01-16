#!/bin/bash
####nohup   ./longjobs.sh > longjobs.out 2>&1 &
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/userhome/code/MinkLoc3D-master
. ~/.bashrc

###different loss weights with triplet 
##master-servant mode：
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal901.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/901.log
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal802.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/802.log
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal703.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/703.log
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal604.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/604.log
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal82.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/82_again.log
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal73.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/73_again.log
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal64.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/64_again.log
 
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal91.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/91.log


# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal82.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/82.log


# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal73.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/73.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal64.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/73.log


# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal46.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/46.log


# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal37.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/37.log


# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal28.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/28.log


# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal19.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/19.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal37.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/37.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/all.log
##10,0,0, all is fused feature 
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal1000.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/1000.log

# ##4,3,3, average case
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal433.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/433.log

# ##5,0,5  image master
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal505.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/505.log

# ##5, 1, 4
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal514.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/514.log

# ## 5, 2,3
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal523.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/523.log


# ##5,5,0 point cloud master
# ## 5,4,1
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal541.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/541.log

# ## 5,3,2
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal532.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/532.log


#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal352.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/352.log


# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB32.txt --model_config ./models/wavemultimodalinfer.txt >wavemultimodal_rgbsketch32.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB41.txt --model_config ./models/wavemultimodalinfer.txt >wavemultimodal_rgbsketch41.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB23.txt --model_config ./models/wavemultimodalinfer.txt >wavemultimodal_rgbsketch23.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB14.txt --model_config ./models/wavemultimodalinfer.txt >wavemultimodal_rgbsketch14.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB253.txt --model_config ./models/wavemultimodalinfer.txt >253.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB613.txt --model_config ./models/wavemultimodalinfer.txt >613.log


# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB361.txt --model_config ./models/wavemultimodalinfer.txt >361.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB370.txt --model_config ./models/wavemultimodalinfer.txt >370.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB460.txt --model_config ./models/wavemultimodalinfer.txt >460.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB631.txt --model_config ./models/wavemultimodalinfer.txt >631.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB424.txt --model_config ./models/wavemultimodalinfer.txt >424.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB451.txt --model_config ./models/wavemultimodalinfer.txt >451.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB415.txt --model_config ./models/wavemultimodalinfer.txt >415.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB433.txt --model_config ./models/wavemultimodalinfer.txt >433.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB442.txt --model_config ./models/wavemultimodalinfer.txt >442.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB343.txt --model_config ./models/wavemultimodalinfer.txt >343.log 

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB334.txt --model_config ./models/wavemultimodalinfer.txt >334.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB811.txt --model_config ./models/wavemultimodalinfer.txt >811.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB721.txt --model_config ./models/wavemultimodalinfer.txt >721.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB712.txt --model_config ./models/wavemultimodalinfer.txt >712.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB613.txt --model_config ./models/wavemultimodalinfer.txt >613.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB352.txt --model_config ./models/wavemultimodalinfer.txt >352.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB154.txt --model_config ./models/wavemultimodalinfer.txt >154.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB901.txt --model_config ./models/wavemultimodalinfer.txt >901.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB244.txt --model_config ./models/wavemultimodalinfer.txt >244.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB622.txt --model_config ./models/wavemultimodalinfer.txt >622.log
