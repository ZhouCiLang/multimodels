#!/bin/bash
####nohup   ./longjobs.sh > longjobs.out 2>&1 &
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/userhome/code/MinkLoc3D-master
. ~/.bashrc



###P
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20231230_1822_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_SVTinfer256_20231230_1822_final.pth >./logs/svt/season/model_SVTinfer256_20231230_1822.txt


# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal352.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20231231_2314_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_SVTinfer256_20231231_2314_final.pth >./logs/svt/season/model_SVTinfer256_20231231_2314_final.txt


# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal433.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20240101_1124_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_SVTinfer256_20240101_1124_final.pth >./logs/svt/season/model_SVTinfer256_20240101_1124_final.txt



# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/extract_feature.py --config ./config/config_baseline_multimodal541.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20240102_0939_final.pth
# /opt/conda/envs/name/bin/python ./robotcar_seasons_benchmark/estimate_season_poses.py --weights ./weights/model_SVTinfer256_20240102_0939_final.pth >./logs/svt/season/model_SVTinfer256_20240102_0939_final.txt


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20231230_1822_final.pth >./logs/svt/kitti/model_SVTinfer256_20231230_1822_final.log
# Dataset: KITTI
# Avg. top 1% recall: 77.38   Avg. similarity: 130.6389   Avg. recall @N:
# [71.42857143 77.38095238 78.57142857 79.76190476 82.14285714 83.33333333
#  83.33333333 83.33333333 85.71428571 85.71428571 86.9047619  86.9047619
#  86.9047619  86.9047619  88.0952381  88.0952381  88.0952381  88.0952381
#  88.0952381  89.28571429 89.28571429 89.28571429 89.28571429 89.28571429
#  89.28571429]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal352.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20231231_2314_final.pth >./logs/svt/kitti/model_SVTinfer256_20231231_2314_final.txt
# Avg. top 1% recall: 76.19   Avg. similarity: 105.1315   Avg. recall @N:
# [75.         76.19047619 76.19047619 78.57142857 80.95238095 82.14285714
#  84.52380952 84.52380952 84.52380952 88.0952381  89.28571429 89.28571429
#  89.28571429 90.47619048 90.47619048 90.47619048 90.47619048 90.47619048
#  90.47619048 90.47619048 91.66666667 91.66666667 91.66666667 92.85714286
#  92.85714286]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal433.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20240101_1124_final.pth >./logs/svt/kitti/model_SVTinfer256_20240101_1124_final.txt
# Avg. top 1% recall: 80.95   Avg. similarity: 104.1296   Avg. recall @N:
# [79.76190476 80.95238095 80.95238095 83.33333333 85.71428571 86.9047619
#  86.9047619  86.9047619  88.0952381  88.0952381  88.0952381  88.0952381
#  88.0952381  88.0952381  88.0952381  88.0952381  88.0952381  88.0952381
#  88.0952381  88.0952381  88.0952381  88.0952381  88.0952381  88.0952381
#  88.0952381 ]

# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal541.txt --model_config ./models/svtinfer256.txt --weights ./weights/model_SVTinfer256_20240102_0939_final.pth >./logs/svt/kitti/model_SVTinfer256_20240102_0939_final.txtAvg. top 1% recall: 75.00   Avg. similarity: 125.9405   Avg. recall @N:
# [73.80952381 75.         79.76190476 83.33333333 83.33333333 83.33333333
#  88.0952381  88.0952381  88.0952381  89.28571429 89.28571429 90.47619048
#  90.47619048 90.47619048 90.47619048 90.47619048 90.47619048 90.47619048
#  90.47619048 90.47619048 90.47619048 90.47619048 90.47619048 90.47619048
#  90.47619048]