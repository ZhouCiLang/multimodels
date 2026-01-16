#!/bin/bash
####nohup   ./longjobs.sh > longjobs.out 2>&1 &
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/userhome/code/MinkLoc3D-master
. ~/.bashrc

###for oxford

###minkloc++
# /opt/conda/envs/name/bin/python ./eval/eval_retrieval.py --config ./config/config_baseline_multimodal.txt --model_config ./models/minklocmultimodal.txt --weights ./weights/model_MinkLocMultimodal_20231128_1008_final.pth >./results/minklocplusplus_oxford_retrieval.txt

# ###ours
# /opt/conda/envs/name/bin/python ./eval/eval_retrieval.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231227_0909_final.pth >./results/ours_oxford_retrieval.txt


###for kitti
####minkloc ++
# /opt/conda/envs/name/bin/python ./eval/eval_kitti_retrieval.py --config ./config/config_baseline_multimodal.txt --model_config ./models/minklocmultimodal.txt --weights ./weights/model_MinkLocMultimodal_20231128_1008_final.pth >./results/minklocplusplus_kitti_retrieval.txt

# ###ours
# /opt/conda/envs/name/bin/python ./eval/eval_kitti_retrieval.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231227_0909_final.pth >./results/ours_kitti_retrieval.txt


#/opt/conda/envs/name/bin/python ./eval/evaluate.py --config ./config/config_baseline.txt --model_config ./models/minkloc3d.txt --weights ./weights/model_MinkLoc3D_20231128_1621_final.pth

#/opt/conda/envs/name/bin/python ./eval/evaluate.py --config ./config/config_baseline_rgb.txt --model_config ./models/minklocrgb.txt --weights ./weights/model_MinkLocRGB_20231129_0329_final.pth


# /opt/conda/envs/name/bin/python ./eval/evaluate.py --config ./config/config_baseline_graph.txt --model_config ./models/minklocmultimodal.txt --weights ./weights/model_MinkLocMultimodal_20231203_1200_final.pth >wave_pc_graphresult.log

# /opt/conda/envs/name/bin/python ./eval/evaluate.py --config ./config/config_baseline_random.txt --model_config ./models/minklocmultimodal.txt --weights ./weights/model_MinkLocMultimodal_20231203_1431_final.pth >pc_random.log
