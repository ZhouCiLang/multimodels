#!/bin/bash
####nohup   ./longjobs.sh > longjobs.out 2>&1 &
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/userhome/code/MinkLoc3D-master
. ~/.bashrc


#################################################################Ablation Start Part#########################################################################################
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231227_1908_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveMultimodalInferAll256_20231227_1908_final.pth >./logs/season/model_WaveMultimodalInferAll256_20231227_1908_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveMultimodalInferAll256_20231227_1908_final.pth  >./logs/season/model_WaveMultimodalInferAll256_20231227_1908_final_5.txt


# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231227_0909_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveMultimodalInferAll256_20231227_0909_final.pth >./logs/season/model_WaveMultimodalInferAll256_20231227_0909_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveMultimodalInferAll256_20231227_0909_final.pth  >./logs/season/model_WaveMultimodalInferAll256_20231227_0909_final_5.txt


# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal.txt --model_config ./models/minklocmultimodalAdd.txt --weights ./weights/model_MinkLocMultimodalAdd_20240105_1518_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_BaseMinkLocMultimodal_20240105_0848_final.pth >./logs/season/model_MinkLocMultimodalAdd_20240105_1518_final.txt


# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal.txt --model_config ./models/basemultimodal.txt --weights ./weights/model_BaseMinkLocMultimodal_20240105_0848_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_BaseMinkLocMultimodal_20240105_0848_final.pth >./logs/season/model_BaseMinkLocMultimodal_20240105_0848_final.txt

/opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py  --config ./config/config_baseline_multimodal442.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240108_1904_final.pth
/opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveMultimodalInferAll256_20240108_1904_final.pth >./logs/season/model_WaveMultimodalInferAll256_20240108_1904.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py  --config ./config/config_baseline_multimodal114.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240108_0955_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveMultimodalInferAll256_20240108_0955_final.pth >./logs/season/model_WaveMultimodalInferAll256_20240108_0955.txt

#################################################################Ablation End Part#########################################################################################


##############################################################SVT#####################################################################################################
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svt.txt --weights ./weights/model_SVT_20231223_1539_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_SVT_20231223_1539_final.pth >./logs/season/model_SVT_20231223_1539_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_SVT_20231223_1539_final.pth  >./logs/season/model_SVT_20231223_1539_final_5.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtqi.txt --weights ./weights/model_SVTQI_20231223_1551_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_SVTQI_20231223_1551_final.pth >./logs/season/model_SVTQI_20231223_1551_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_SVTQI_20231223_1551_final.pth  >./logs/season/model_SVTQI_20231223_1551_final_5.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtinferall128.txt --weights ./weights/model_SVTinferall128_20231224_0021_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_SVTinferall128_20231224_0021_final.pth >./logs/season/model_SVTinferall128_20231224_0021_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_SVTinferall128_20231224_0021_final.pth  >./logs/season/model_SVTinferall128_20231224_0021_final_5.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtqiinferall128.txt --weights ./weights/model_SVTQIinferall128_20231224_0029_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_SVTQIinferall128_20231224_0029_final.pth >./logs/season/model_SVTQIinferall128_20231224_0029_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_SVTQIinferall128_20231224_0029_final.pth  >./logs/season/model_SVTQIinferall128_20231224_0029_final_5.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtqiinferall256.txt --weights ./weights/model_SVTQIinferall256_20231224_0920_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_SVTQIinferall256_20231224_0920_final.pth >./logs/season/model_SVTQIinferall256_20231224_0920_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_SVTQIinferall256_20231224_0920_final.pth  >./logs/season/model_SVTQIinferall256_20231224_0920_final_5.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtinferall256.txt --weights ./weights/model_SVTinferall256_20231224_0900_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_SVTinferall256_20231224_0900_final.pth >./logs/season/model_SVTinferall256_20231224_0900_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_SVTinferall256_20231224_0900_final.pth  >./logs/season/model_SVTinferall256_20231224_0900_final_5.txt


########################################################################################PCS +RGB#####################################################################################
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal352.txt --model_config ./models/wavemultimodalinferall.txt --weights ./weights/model_WaveMultimodalInferAll_20231226_0835_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveMultimodalInferAll_20231226_0835_final.pth >./logs/season/model_WaveMultimodalInferAll_20231226_0835_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveMultimodalInferAll_20231226_0835_final.pth  >./logs/season/model_WaveMultimodalInferAll_20231226_0835_final_5.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal352.txt --model_config ./models/wavemultimodalinfer.txt --weights ./weights/model_WaveMultimodalInfer_20231225_1018_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveMultimodalInfer_20231225_1018_final.pth >./logs/season/model_WaveMultimodalInfer_20231225_1018_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveMultimodalInfer_20231225_1018_final.pth  >./logs/season/model_WaveMultimodalInfer_20231225_1018_final_5.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231223_0949_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveMultimodalInferAll256_20231223_0949_final.pth >./logs/season/model_WaveMultimodalInferAll256_20231223_0949_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveMultimodalInferAll256_20231223_0949_final.pth  >./logs/season/model_WaveMultimodalInferAll256_20231223_0949_final_5.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal352.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231224_1033_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveMultimodalInferAll256_20231224_1033_final.pth >./logs/season/model_WaveMultimodalInferAll256_20231224_1033_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveMultimodalInferAll256_20231224_1033_final.pth  >./logs/season/model_WaveMultimodalInferAll256_20231224_1033_final_5.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal256infer.txt --weights ./weights/model_WaveMultimodalInfer256_20231224_1031_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveMultimodalInfer256_20231224_1031_final.pth >./logs/season/model_WaveMultimodalInfer256_20231224_1031_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveMultimodalInfer256_20231224_1031_final.pth  >./logs/season/model_WaveMultimodalInfer256_20231224_1031_final_5.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal352.txt --model_config ./models/wavemultimodal256infer.txt --weights ./weights/model_WaveMultimodalInfer256_20231224_1531_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveMultimodalInfer256_20231224_1531_final.pth >./logs/season/model_WaveMultimodalInfer256_20231224_1531_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveMultimodalInfer256_20231224_1531_final.pth  >./logs/season/model_WaveMultimodalInfer256_20231224_1531_final_5.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinfer.txt --weights ./weights/model_WaveMultimodal_20231201_1443_final.pth --feature_path ./robotcar_seasons_benchmark/season_scan_embeddings_multi_pc_org1.pickle
#python ./robotcar_seasons_benchmark/estimate_season_poses.py >./robotcar_seasons_benchmark/5m_400.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveMultimodal_20231201_1443_final.pth >./logs/season/model_WaveMultimodal_20231201_1443_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveMultimodal_20231201_1443_final.pth  >./logs/season/model_WaveMultimodal_20231201_1443_final_5.txt


###############################################################################PCS +RGB Sketches #####################################################################################################
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/wavemultimodalinfer.txt --weights ./weights/model_WaveMultimodalInfer_20231225_2005_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveMultimodalInfer_20231225_1008_final.pth >./logs/season/model_WaveMultimodalInfer_20231225_2005_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveMultimodalInfer_20231225_2005_final.pth  >./logs/season/model_WaveMultimodalInfer_20231225_2005_final_5.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal_sketchRGB352.txt --model_config ./models/wavemultimodalinfer.txt --weights ./weights/model_WaveMultimodalInfer_20231225_1008_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveMultimodalInfer_20231225_1008_final.pth >./logs/season/model_WaveMultimodalInfer_20231225_1008_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveMultimodalInfer_20231225_1008_final.pth  >./logs/season/model_WaveMultimodalInfer_20231225_1008_final_5.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall.txt --weights ./weights/model_WaveMultimodalInferAll_20231222_2043_final.pth 

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveMultimodalInferAll_20231222_2043_final.pth >./logs/season/model_WaveMultimodalInferAll_20231222_2043_final.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveMultimodalInferAll_20231222_2043_final.pth  >./logs/season/model_WaveMultimodalInferAll_20231222_2043_final_5.txt


# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231223_0950_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveMultimodalInferAll256_20231223_0950_final.pth >./logs/season/model_WaveMultimodalInferAll256_20231223_0950_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveMultimodalInferAll256_20231223_0950_final.pth  >./logs/season/model_WaveMultimodalInferAll256_20231223_0950_final_5.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal_sketchRGB352.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231223_1601_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveMultimodalInferAll256_20231223_1601_final.pth >./logs/season/model_WaveMultimodalInferAll256_20231223_1601_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveMultimodalInferAll256_20231223_1601_final.pth  >./logs/season/model_WaveMultimodalInferAll256_20231223_1601_final_5.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/wavemultimodal256infer.txt --weights ./weights/model_WaveMultimodalInfer256_20231224_1734_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveMultimodalInfer256_20231224_1734_final.pth >./logs/season/model_WaveMultimodalInfer256_20231224_1734_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveMultimodalInfer256_20231224_1734_final.pth  >./logs/season/model_WaveMultimodalInfer256_20231224_1734_final_5.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal_sketchRGB352.txt --model_config ./models/wavemultimodal256infer.txt --weights ./weights/model_WaveMultimodalInfer256_20231224_2346_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveMultimodalInfer256_20231224_2346_final.pth >./logs/season/model_WaveMultimodalInfer256_20231224_2346_final.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveMultimodalInfer256_20231224_2346_final.pth  >./logs/season/model_WaveMultimodalInfer256_20231224_2346_final_5.txt


############################################################################PCS ###################################################################################### 
#128D
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline.txt --model_config ./models/wave3d.txt --weights ./weights/model_Wave3D_20231206_1503_final.pth 

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_Wave3D_20231206_1503_final.pth >./logs/season/model_Wave3D_20231206_1503_final.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_Wave3D_20231206_1503_final.pth  >./logs/season/model_Wave3D_20231206_1503_final_5.txt


###PCS 256D
#/opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline.txt --model_config ./models/wave3d.txt --weights ./weights/model_Wave3D_20231128_1620_final.pth


########################################################################## RGB ########################################################################################
###RGB 128D
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergb.txt --weights ./weights/model_WaveRGB_20231204_1649_final.pth 

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveRGB_20231204_1649_final.pth >./logs/season/model_WaveRGB_20231204_1649_final.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveRGB_20231204_1649_final.pth  >./logs/season/model_WaveRGB_20231204_1649_final_5.txt

###RGB 256D
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergb256.txt --weights ./weights/model_WaveRGB_20231206_1509_final.pth 

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveRGB_20231129_0054_final.pth >./logs/season/model_WaveRGB_20231206_1509.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveRGB_20231129_0054_final.pth  >./logs/season/model_WaveRGB_20231206_1509_5.txt


#######################################################################MinkLoc3d++ #######################################################################################
#/opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal.txt --model_config ./models/minklocmultimodal.txt --weights ./weights/minklocmultimodal_baseline.pth

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline.txt --model_config ./models/minkloc3d.txt --weights ./weights/minkloc3d_baseline.pth

#/opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline.txt --model_config ./models/minkloc3d.txt --weights ./weights/model_MinkLoc3D_20231128_1621_final.pth

###RGB
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_rgb.txt --model_config ./models/minklocrgb.txt --weights ./weights/model_WaveRGB_20231129_0054_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_WaveRGB_20231129_0054_final.pth >./logs/season/model_WaveRGB_20231129_0054_final.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_WaveRGB_20231129_0054_final.pth  >./logs/season/model_WaveRGB_20231129_0054_final_5.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal.txt --model_config ./models/minklocmultimodal.txt --weights ./weights/minklocmultimodal_baseline.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/minklocmultimodal_baseline.pth >./logs/season/minklocmultimodal_baseline.txt
#/opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/minklocmultimodal_baseline.pth  >./logs/season/minklocmultimodal_baseline_5.txt


# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal.txt --model_config ./models/minklocmultimodal.txt --weights ./weights/model_MinkLocMultimodal_20231128_1008_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_MinkLocMultimodal_20231128_1008_final.pth >./logs/season/model_MinkLocMultimodal_20231128_1008.txt
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --threshold 5 --weights ./weights/model_MinkLocMultimodal_20231128_1008_final.pth  >./logs/season/model_MinkLocMultimodal_20231128_1008_5.txt