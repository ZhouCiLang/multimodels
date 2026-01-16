#!/bin/bash
####nohup   ./longjobs.sh > longjobs.out 2>&1 &
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/userhome/code/MinkLoc3D-master
. ~/.bashrc

##svt
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB32.txt --model_config ./models/svtinfer128.txt >./logs/svt/32.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB41.txt --model_config ./models/svtinfer128.txt >./logs/svt/41.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB23.txt --model_config ./models/svtinfer128.txt >./logs/svt/23.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB14.txt --model_config ./models/svtinfer128.txt >./logs/svt/14.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB253.txt --model_config ./models/svtinfer128.txt >./logs/svt/253.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB613.txt --model_config ./models/svtinfer128.txt >./logs/svt/613.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB361.txt --model_config ./models/svtinfer128.txt >./logs/svt/361.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB370.txt --model_config ./models/svtinfer128.txt >./logs/svt/370.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB460.txt --model_config ./models/svtinfer128.txt >./logs/svt/460.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB631.txt --model_config ./models/svtinfer128.txt >./logs/svt/631.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB424.txt --model_config ./models/svtinfer128.txt >./logs/svt/424.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB451.txt --model_config ./models/svtinfer128.txt >./logs/svt/451.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB415.txt --model_config ./models/svtinfer128.txt >./logs/svt/415.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB433.txt --model_config ./models/svtinfer128.txt >./logs/svt/433.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB442.txt --model_config ./models/svtinfer128.txt >./logs/svt/442.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB343.txt --model_config ./models/svtinfer128.txt >./logs/svt/343.log 

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB334.txt --model_config ./models/svtinfer128.txt >./logs/svt/334.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB811.txt --model_config ./models/svtinfer128.txt >./logs/svt/811.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB721.txt --model_config ./models/svtinfer128.txt >./logs/svt/721.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB712.txt --model_config ./models/svtinfer128.txt >./logs/svt/712.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB613.txt --model_config ./models/svtinfer128.txt >./logs/svt/613.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB352.txt --model_config ./models/svtinfer128.txt >./logs/svt/352.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB154.txt --model_config ./models/svtinfer128.txt >./logs/svt/154.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB1000.txt --model_config ./models/svtinfer128.txt >./logs/svt/100.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB244.txt --model_config ./models/svtinfer128.txt >./logs/svt/244.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB622.txt --model_config ./models/svtinfer128.txt >./logs/svt/622.log


###different loss

#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_contrast.txt --model_config ./models/wavemultimodalinferall256.txt>./logs/ablations/wavemultimodalinferall256_contrast.log

#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_ap.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/waveminklocmultimodalinferall256_tap.log

##wait to run for sketches?


# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_contrast.txt --model_config ./models/wavemultimodalinfer.txt>wavemultimodalinfer_contrast.log
# /opt/conda/envs/name/bin/python ./training/train_ap.py --config ./config/config_baseline_multimodal_ap1.txt --model_config ./models/wavemultimodalinfer.txt 

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_ap.txt --model_config ./models/wavemultimodalinfer.txt > waveminklocmultimodalinfer_tap.log

#python ./eval/evaluate.py --config ./config/config_baseline_multimodal_ap.txt --model_config ./models/minklocmultimodal.txt --weights ./weights/model_MinkLocMultimodal_20231203_1658_final.pth

##no wave AP
#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_ap.txt --model_config ./models/minklocmultimodal.txt > minklocmultimodal_tap.log
##model_MinkLocMultimodal_20231203_1658_final.pth



###different loss weights with triplet 
#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal352.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/352.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB32.txt --model_config ./models/wavemultimodalinfer.txt >wavemultimodal_rgbsketch32.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB41.txt --model_config ./models/wavemultimodalinfer.txt >wavemultimodal_rgbsketch41.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB23.txt --model_config ./models/wavemultimodalinfer.txt >wavemultimodal_rgbsketch23.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB14.txt --model_config ./models/wavemultimodalinfer.txt >wavemultimodal_rgbsketch14.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB253.txt --model_config ./models/wavemultimodalinfer.txt >253.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB613.txt --model_config ./models/wavemultimodalinfer.txt >613.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGBnormalize.txt --model_config ./models/wavemultimodalinfer.txt >wavemultimodal_rgbsketch_norm.log

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
