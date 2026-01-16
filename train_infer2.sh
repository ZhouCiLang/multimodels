#!/bin/bash
####nohup   ./longjobs.sh > longjobs.out 2>&1 &
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/userhome/code/MinkLoc3D-master
. ~/.bashrc

/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal442.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/442_again.log

/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal64.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/64_ag.log
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal111.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/111.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal110.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/110.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal119.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/119.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal118.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/118.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal117.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/117.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal116.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/116.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal115.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/115.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal114.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/114.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal113.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/113.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal112.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/112.log



#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/all/nobasicblock.log


#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_rgb.txt --model_config ./models/minklocrgb.txt >./logs/ablations/base_rgb.log

#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/basemultimodal.txt >./logs/ablations/basemultimodal.log


#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergbtest.txt >./logs/ablations/wavergb34_256.log

#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/wavemultimodalinferall256_1again.log

