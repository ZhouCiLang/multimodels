#!/bin/bash
####nohup   ./longjobs.sh > longjobs.out 2>&1 &
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/userhome/code/MinkLoc3D-master
. ~/.bashrc

##test svt series:
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/svt.txt >./logs/svt_sketchRGB.log

#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/svtadd256.txt >./logs/svtinferall256_sketchRGB_add.txt

/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/svtcon256.txt >./logs/svtinferall256_sketchRGB_con.txt

#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/svtinfer256.txt >./logs/svtinferall256_sketchRGB_image.txt

#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/svtinfer128.txt >./logs/svtinferall128_sketchRGB.txt


# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/svtqi.txt >./logs/svtqi_sketchRGB.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/svtqiinfer256.txt >./logs/svtqi_inferall256_sketchRGB.txt

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/svtqiinfer128.txt >./logs/svtqi_inferall128_sketcthRGB.txt


# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svt.txt >./logs/svt.txt

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtinferall128.txt >./logs/svtinferall128.txt

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtinferall256.txt >./logs/svtinferall256.txt

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtqi.txt >./logs/svtqi.txt

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtqiinferall128.txt >./logs/svtqiinferall128.txt

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtqiinferall256.txt >./logs/svtqiinferall256.txt


