#!/bin/bash
####nohup   ./longjobs.sh > longjobs.out 2>&1 &
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/userhome/code/MinkLoc3D-master
. ~/.bashrc

/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal622.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/622.log
/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal613.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/613.log
/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal442.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/442.log
/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal424.txt --model_config ./models/wavemultimodalinferall256.txt >./logs/ablations/424.log


# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_rgb_sketch.txt --model_config ./models/wavergb.txt >wavergb_sketch.log

# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/wavemultimodalinfer.txt 
# #model_WaveMultimodalInfer_20231212_0938_final
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/wavemultimodal256infer.txt > wavemultimodalinfer256_RGBsketch.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/minklocmultimodal.txt > minklocmultimodal_RGBsketch.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/minklocmultimodal128.txt > minklocmultimodal128_RGBsketch.log

# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketch_random.txt --model_config ./models/wavemultimodalinfer.txt > wavemultimodalinfer_sketch_random.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketch_random.txt --model_config ./models/wavemultimodal256infer.txt > wavemultimodalinfer256_sketch_random.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketch_random.txt --model_config ./models/minklocmultimodal.txt > minklocmultimodal_sketch_random.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketch_random.txt --model_config ./models/minklocmultimodal128.txt > minklocmultimodal128_sketch_random.log

# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketch_nonuniform.txt --model_config ./models/wavemultimodalinfer.txt > wavemultimodalinfer_sketch_uniform.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketch_nonuniform.txt --model_config ./models/wavemultimodal256infer.txt > wavemultimodalinfer256_sketch_uniform.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketch_nonuniform.txt --model_config ./models/minklocmultimodal.txt > minklocmultimodal_sketch_uniform.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketch_nonuniform.txt --model_config ./models/minklocmultimodal128.txt > minklocmultimodal128_sketch_uniform.log

#model_WaveMultimodalInfer_20231212_0938_final
# Avg. top 1% recall: 98.70  Avg. top 1 recall: 95.09   Avg. similarity: 63.5590   Avg. recall @N:
# [95.08827091 97.5374933  98.37437392 98.71808517 98.95380249 99.1120663
#  99.20386661 99.30973453 99.39127735 99.45679403 99.51995291 99.55553794
#  99.59491002 99.64186455 99.67077092 99.69468814 99.7162739  99.73169842
#  99.75871598 99.77908818 99.79110115 99.803716   99.81411019 99.82350151
#  99.83897308]
 
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergbtest.txt> wavergb_34_output128.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_rgb.txt --model_config ./models/wavergb.txt> wavergb18_output256.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinfer.txt>wavemultimodalinfer2.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net2.txt > wavemultimodal_net2_again.log

# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_pc_graph.txt --model_config ./models/wave3d.txt >wave3donly_graph.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_pc_random.txt --model_config ./models/wave3d.txt >wave3donly_random.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_pc_nonuniform.txt --model_config ./models/wave3d.txt >wave3donly_nonuniform.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_pc_graph.txt --model_config ./models/minkloc3d.txt >mink3donly_graph128.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_pc_random.txt --model_config ./models/minkloc3d.txt >mink3donly_random128.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_pc_nonuniform.txt --model_config ./models/minkloc3d.txt >mink3donly_nonuniform128.log


#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergb.txt> wavergb_18_output128.log



##wave image without amplitude normalization
#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal.txt>wavemultimodal.log
##OUTPUT:model_WaveMultimodal_20231128_1128_final.pth

## wave 
#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline.txt --model_config ./models/wave3d.txt>wave3d.log
##output:model_Wave3D_20231128_1620_final.pth

##wave image without amplitude normalization
#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergb.txt> wavergb.log
##OUTPUT: model_WaveRGB_20231129_0054_final.pth

##wave image with amplitude normalization
#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal.txt>wavemultimodal1.log
##OUTPUT:model_WaveMultimodal_20231201_1443_final.pth


##wave image with amplitude normalization
#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergb.txt> wavergb1.log
##OUTPUT: model_WaveRGB_20231201_2142_final.pth

#python ./eval/evaluate.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal.txt --weights ./weights/model_WaveMultimodal_20231128_1128_final.pth
#python ./eval/evaluate.py --config ./config/config_baseline.txt --model_config ./models/wave3d.txt --weights ./weights/model_Wave3D_20231128_1620_final.pth
#python ./eval/evaluate.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergb.txt --weights ./weights/model_WaveRGB_20231129_0054_final.pth

#python ./eval/evaluate.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal.txt --weights ./weights/model_WaveMultimodal_20231201_1443_final.pth
#python ./eval/evaluate.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergb.txt --weights ./weights/model_WaveRGB_20231201_2142_final.pth


# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_random.txt --model_config ./models/wavemultimodal256infer.txt > wavemultinfer_random256.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_nonuniform.txt --model_config ./models/wavemultimodal256infer.txt > wavemultinfer_nonuniform256.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_random.txt --model_config ./models/minklocmultimodal.txt > minklocmultimodal_random.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_nonuniform.txt --model_config ./models/minklocmultimodal.txt> minklocmultimodal_nouniform.log

#/root/miniconda3/envs/torch1.9/bin/python ./eval/evaluate.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/wavemultimodalinfer.txt --weights ./weights/model_WaveMultimodalInfer_20231212_0938_final.pth


# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketch.txt --model_config ./models/wavemultimodalinfer.txt > wavemultimodalinfer_sketch.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketch.txt --model_config ./models/wavemultimodal256infer.txt > wavemultimodalinfer256_sketch.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketch.txt --model_config ./models/minklocmultimodal.txt > minklocmultimodal_sketch.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketch.txt --model_config ./models/minklocmultimodal128.txt > minklocmultimodal128_sketch.log



# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_pc_graph.txt --model_config ./models/wave3d.txt >wave3donly_graph_256.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_pc_random.txt --model_config ./models/wave3d.txt >wave3donly_random_256.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_pc_nonuniform.txt --model_config ./models/wave3d.txt >wave3donly_nonuniform_256.log

#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinfer.txt > wavemultimodalinfer_second.txt

# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketch_grid.txt --model_config ./models/wavemultimodalinfer.txt

#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_sketch_nonuniform.txt --model_config ./models/wavemultimodalinfer.txt >16nouniform_multi.log



# #wave skectch-PCs,output 128d
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_graph.txt --model_config ./models/wavemultimodalinfer.txt > wavemultinfer_pc_graph_infer.log
# #
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_random.txt --model_config ./models/wavemultimodalinfer.txt > wavemultinfer_pc_random_infer.log
# # 
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_nonuniform.txt --model_config ./models/wavemultimodalinfer.txt > wavemultinfer_pc_nonun_infer.log

# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_graph.txt --model_config ./models/wavemultimodal.txt > wavemulti_pc_graph1.log
# #
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_random.txt --model_config ./models/wavemultimodal.txt > wavemulti_pc_randomnew1.log
# # #
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_nonuniform.txt --model_config ./models/wavemultimodal.txt > wavemulti_pc_nonuniform1.log
# #


##nowave
#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/minklocmultimodal.txt

#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline.txt --model_config ./models/minkloc3d.txt

#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_rgb.txt --model_config ./models/minklocrgb.txt


#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline.txt --model_config ./models/wave3d.txt

# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergb.txt



#python ./eval/evaluate.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinfer.txt --weights ./weights/model_WaveMultimodal_20231201_1443_final.pth
##Avg. top 1% recall: 95.59  Avg. top 1 recall: 87.41   Avg. similarity: 122.388

# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_rgb_sketch.txt --model_config ./models/wavergb256.txt >wavergb256_sketch.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_rgb_sketch.txt --model_config ./models/minklocrgb128.txt >minklocrgb128_sketch.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_rgb_sketch.txt --model_config ./models/minklocrgb.txt >minklocrgb_sketch.log

# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_graph.txt --model_config ./models/minklocmultimodal128.txt > minklocmultimodal_graph128.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_random.txt --model_config ./models/minklocmultimodal128.txt > minklocmultimodal_random128.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_nonuniform.txt --model_config ./models/minklocmultimodal128.txt> minklocmultimodal_nouniform128.log

#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_graph.txt --model_config ./models/wavemultimodal256infer.txt > wavemultinfer_graph256.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_graph.txt --model_config ./models/minklocmultimodal.txt > minklocmultimodal_graph.log


# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_random.txt --model_config ./models/wavemultimodal256infer.txt > wavemultinfer_random256.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_nonuniform.txt --model_config ./models/wavemultimodal256infer.txt > wavemultinfer_nonuniform256.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_random.txt --model_config ./models/minklocmultimodal.txt > minklocmultimodal_random.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_nonuniform.txt --model_config ./models/minklocmultimodal.txt> minklocmultimodal_nouniform.log


#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_ap.txt --model_config ./models/minkloc3d.txt 
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal_ap1.txt --model_config ./models/wavemultimodalinfer.txt 

# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline.txt --model_config ./models/minkloc3d.txt > minkloc3d_128.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline.txt --model_config ./models/wave3d.txt>wave3d_128.log
# # /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinfer.txt > minklocmultimodal128_mlp.log
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal256infer.txt > wavemultimodal256infer.log

##skectch-PCs
#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_graph.txt --model_config ./models/minklocmultimodal.txt > wave_pc_graph.log
##output:model_MinkLocMultimodal_20231203_1200_final.pth

#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_random.txt --model_config ./models/minklocmultimodal.txt > wave_pc_random.log
##output:model_MinkLocMultimodal_20231203_1431_final.pth

#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_nonuniform.txt --model_config ./models/minklocmultimodal.txt > wave_pc_nonuniform.log
##output: model_MinkLocMultimodal_20231203_1700_final。pth

##nowave
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_multimodal.txt --model_config ./models/minklocmultimodal.txt > minklocmultimodal.log

# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline.txt --model_config ./models/minkloc3d.txt > minkloc3d.log

# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_rgb.txt --model_config ./models/minklocrgb.txt >minklocrgb.log


#/root/miniconda3/envs/torch1.9/bin/python ./eval/evaluate.py --config ./config/config_baseline_multimodal.txt --model_config ./models/minklocmultimodal.txt --weights ./weights/model_MinkLocMultimodal_20231128_1008_final.pth
#Avg. top 1% recall: 98.97  Avg. top 1 recall: 96.28   Avg. similarity: 77.0385

#/root/miniconda3/envs/torch1.9/bin/python ./eval/evaluate.py --config ./config/config_baseline.txt --model_config ./models/minkloc3d.txt --weights ./weights/model_MinkLoc3D_20231128_1621_final.pth

#/root/miniconda3/envs/torch1.9/bin/python ./eval/evaluate.py --config ./config/config_baseline_rgb.txt --model_config ./models/minklocrgb.txt --weights ./weights/model_MinkLocRGB_20231129_0329_final.pth


# /root/miniconda3/envs/torch1.9/bin/python ./eval/evaluate.py --config ./config/config_baseline_graph.txt --model_config ./models/minklocmultimodal.txt --weights ./weights/model_MinkLocMultimodal_20231203_1200_final.pth >wave_pc_graphresult.log

# /root/miniconda3/envs/torch1.9/bin/python ./eval/evaluate.py --config ./config/config_baseline_random.txt --model_config ./models/minklocmultimodal.txt --weights ./weights/model_MinkLocMultimodal_20231203_1431_final.pth >pc_random.log

#wave skectch-PCs, output 256d, concate
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_graph.txt --model_config ./models/wavemultimodal.txt > wavemulti_pc_graph.log
# #
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_random.txt --model_config ./models/wavemultimodal.txt > wavemulti_pc_randomnew.log
# # #
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_nonuniform.txt --model_config ./models/wavemultimodal.txt > wavemulti_pc_nonuniform.log
# # #

#wave skectch-PCs,output 128d
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_graph.txt --model_config ./models/wavemultimodalinfer.txt > wavemultinfer_pc_graph.log
# #
# /root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_random.txt --model_config ./models/wavemultimodalinfer.txt > wavemultinfer_pc_randomnew.log
# #
#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_nonuniform.txt --model_config ./models/wavemultimodalinfer.txt > wavemultinfer_pc_nonuniform.log

#/root/miniconda3/envs/torch1.9/bin/python ./training/train.py --config ./config/config_baseline_graph.txt --model_config ./models/wavemultimodalinfer.txt