#!/bin/bash
####nohup   ./longjobs.sh > longjobs.out 2>&1 &
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/userhome/code/MinkLoc3D-master
. ~/.bashrc


/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal811.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/811.log
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal732.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/732.log
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal721.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/723.log
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal721.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/721.log
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal631.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/631.log

# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal622.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/622.log
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal613.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/613.log
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal442.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/442.log
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal424.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/424.log





##no wave with add

#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/minklocmultimodalAdd.txt >./logs/ablations/nowave_con.log


##wave with add
#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodaladd.txt >wavemultimodaladd256.log

##wave with add
#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodaladd.txt >wavemultimodaladd.log
##Output:model_WaveMultimodalAdd_20231203_0952_final

##wave with concat and mlp
#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalconcat.txt >wavemultimodalconcatmlp_128.log
##model_WaveMultimodalConcat128_20231203_1716_final

##wave with concat and fc
#/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalconcat.txt >wavemultimodalconcatfc_128.log
##Output:model_WaveMultimodalConcat128_20231203_1123_final