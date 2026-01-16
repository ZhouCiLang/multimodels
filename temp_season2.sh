#!/bin/bash
####nohup   ./longjobs.sh > longjobs.out 2>&1 &
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/userhome/code/MinkLoc3D-master
. ~/.bashrc

/opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/svtadd256.txt --weights ./weights/model_SVTadd256_20240112_1628_final.pth >./logs/svt/kitti/model_SVTadd256_20240112_1628_final.txt


/opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/svtcon256.txt --weights ./weights/model_SVTcon256_20240112_1633_final.pth >./logs/svt/kitti/model_SVTcon256_20240112_1633_final.txt



# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal_sketchRGB451.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20240112_1155_final.pth >./logs/svt/kitti/model_SVTinfer256_20240112_1155_final.txt


# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20240112_1155_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_SVTinfer256_20240112_1155_final.pth >./logs/svt/season/model_SVTinfer256_20240112_1155.txt

# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal_sketchRGB451.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20240102_1802_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_SVTinfer256_20240102_1802_final.pth >./logs/svt/season/model_SVTinfer256_20240102_1802_final.txt


# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20231231_0317_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_SVTinfer256_20231231_0317_final.pth >./logs/svt/season/model_SVTinfer256_20231231_0317.txt


# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal_sketchRGB361.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20240101_1647_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_SVTinfer256_20240101_1647_final.pth >./logs/svt/season/model_SVTinfer256_20240101_1647_final.txt


# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal_sketchRGB433.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20240103_0320_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_SVTinfer256_20240103_0320_final.pth >./logs/svt/season/model_SVTinfer256_20240103_0320_final.txt

# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal_sketchRGB433.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20240103_0320_final.pth >./logs/svt/kitti/model_SVTinfer256_20240103_0320_final.txt


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal_sketchRGB451.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20240102_1802_final.pth >./logs/svt/kitti/model_SVTinfer256_20240102_1802_final.txt
# Avg. top 1% recall: 77.38   Avg. similarity: 130.6389   Avg. recall @N:
# [71.42857143 77.38095238 78.57142857 79.76190476 82.14285714 83.33333333
#  83.33333333 83.33333333 85.71428571 85.71428571 86.9047619  86.9047619
#  86.9047619  86.9047619  88.0952381  88.0952381  88.0952381  88.0952381
#  88.0952381  89.28571429 89.28571429 89.28571429 89.28571429 89.28571429
#  89.28571429]



# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20231231_0317_final.pth >./logs/svt/kitti/model_SVTinfer256_20231231_0317_final.txt
# Avg. top 1% recall: 82.14   Avg. similarity: 98.0387   Avg. recall @N:
# [79.76190476 82.14285714 85.71428571 86.9047619  88.0952381  88.0952381
#  89.28571429 89.28571429 89.28571429 89.28571429 89.28571429 89.28571429
#  89.28571429 89.28571429 89.28571429 89.28571429 89.28571429 89.28571429
#  89.28571429 89.28571429 89.28571429 89.28571429 89.28571429 89.28571429
#  89.28571429]



# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal_sketchRGB361.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20240101_1647_final.pth >./logs/svt/kitti/model_SVTinfer256_20240101_1647_final.txt
# Avg. top 1% recall: 77.38   Avg. similarity: 115.9788   Avg. recall @N:
# [72.61904762 77.38095238 83.33333333 88.0952381  89.28571429 90.47619048
#  90.47619048 90.47619048 90.47619048 90.47619048 90.47619048 90.47619048
#  90.47619048 90.47619048 90.47619048 90.47619048 90.47619048 91.66666667
#  92.85714286 92.85714286 92.85714286 92.85714286 92.85714286 92.85714286
#  92.85714286]
