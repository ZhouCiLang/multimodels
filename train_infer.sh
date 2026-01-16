#!/bin/bash
####nohup   ./longjobs.sh > longjobs.out 2>&1 &
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/userhome/code/MinkLoc3D-master
. ~/.bashrc

/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal115.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/115.log

/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal111.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/111_0.1.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal114.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/114.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal113.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/113.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal112.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/112.log


# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal37.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/37.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/all.log


#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/basemultimodal.txt >./logs/ablations/basemultimodal1.log
#####inferall
#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal352.txt --model_config ./models/wavemultimodalinferall.txt >./logs/wavemultimodalinferall.log


# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall.txt >infer37.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal352.txt --model_config ./models/wavemultimodalinferall.txt >./logs/wavemultimodalinferall.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall256.txt >wavemultimodalinferall256.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal352.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/wavemultimodalinferall256_352.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/wavemultimodalinferall256.txt >wavemultimodalinferall256_PCSRGBs.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB352.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/wavemultimodalinferall256_PCSRGBs352.log


###infer
#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal352.txt --model_config ./models/wavemultimodalinfer.txt >./logs/wavemultimodalinfer_352.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal256infer.txt >./logs/wavemultimodalinfer256_again.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal352.txt --model_config ./models/wavemultimodal256infer.txt >./logs/wavemultimodalinfer256_352.log


#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/wavemultimodalinfer.txt >wavemultimodalinfer_PCSRGBs.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB352.txt --model_config ./models/wavemultimodalinfer.txt >./logs/wavemultimodalinfer_PCSRGBs352.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/wavemultimodal256infer.txt >./logs/wavemultimodalinfer256_PCSRGBs.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB352.txt --model_config ./models/wavemultimodal256infer.txt >./logs/wavemultimodalinfer256_PCSRGBs352.log


##wave with infer V0
#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinfer.txt >wavemultimodalinfer.log
##Output: model_WaveMultimodal_20231201_1443_final.pth

##wave with infer V1
#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinfer.txt >wavemultimodalinfer1.log
##Output:model_WaveMultimodalInfer_20231202_1229_final.pth
