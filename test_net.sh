#!/bin/bash
####nohup   ./longjobs.sh > longjobs.out 2>&1 &
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/userhome/code/MinkLoc3D-master
. ~/.bashrc

##net tests
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net1.txt >./logs/ablations/net1.log
##
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net2.txt >./logs/ablations/net2.log
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net3.txt >./logs/ablations/net3.log
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net4.txt >./logs/ablations/net4.log
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net5.txt >./logs/ablations/net5.log
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net6.txt >./logs/ablations/net6.log
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_minkloc3dv2net.txt >./logs/net7.log
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_minkloc3dv2net1.txt >./logs/ablations/net8.log
/opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_minkloc3dv2net2.txt >./logs/ablations/net9.log



# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net1.txt > wavemultimodal_net1.log
# ##
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net2.txt > wavemultimodal_net2.log
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net3.txt > wavemultimodal_net3.log
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net4.txt > wavemultimodal_net4.log
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net5.txt > wavemultimodal_net5.log
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net6.txt > wavemultimodal_net6.log
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_minkloc3dv2net.txt > wavemultimodal_minkloc3dv2net.log
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_minkloc3dv2net1.txt > wavemultimodal_minkloc3dv2net1.log
# /opt/conda/envs/name/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_minkloc3dv2net2.txt > wavemultimodal_minkloc3dv2net2.log
