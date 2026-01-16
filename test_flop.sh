#!/bin/bash
####nohup   ./longjobs.sh > longjobs.out 2>&1 &
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/userhome/code/MinkLoc3D-master
. ~/.bashrc

/opt/conda/envs/name/bin/python ./eval/eval_time.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20231231_0317_final.pth >our_svt_flop.log

##Test FLOPs and time
# /opt/conda/envs/name/bin/python ./eval/eval_time.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinfer.txt --weights ./weights/model_WaveMultimodalInfer_20231206_1914_final.pth >our_flop.log
# /opt/conda/envs/name/bin/python ./eval/eval_time.py --config ./config/config_baseline.txt --model_config ./models/wave3d.txt --weights ./weights/model_Wave3D_20231206_1503_final.pth >our_point_flop.log

#/opt/conda/envs/name/bin/python ./eval/eval_time.py --config ./config/config_baseline_multimodal_sketch.txt --model_config ./models/wavemultimodalinfer.txt --weights ./weights/model_WaveMultimodalInfer_20231212_1006_final.pth >our_flop.log
#/opt/conda/envs/name/bin/python ./eval/eval_time.py --config ./config/config_pc_graph.txt --model_config ./models/wave3d.txt --weights ./weights/model_Wave3D_20231207_1050_final.pth >our_point_flop.log
#/opt/conda/envs/name/bin/python ./eval/eval_time.py --config ./config/config_baseline_multimodal.txt --model_config ./models/minklocmultimodal.txt --weights ./weights/minklocmultimodal_baseline.pth >minkloc_flop.log
#/opt/conda/envs/name/bin/python ./eval/eval_time.py --config ./config/config_baseline.txt --model_config ./models/minkloc3d.txt --weights ./weights/minkloc3d_baseline.pth >minkloc_point_flop.log


#/opt/conda/envs/name/bin/python ./eval/eval_time.py --config ./config/config_baseline_multimodal_sketch.txt --model_config ./models/minklocmultimodal.txt --weights ./weights/model_MinkLocMultimodal_20231213_0328_final.pth