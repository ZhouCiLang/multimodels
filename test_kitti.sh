i#!/bin/bash
####nohup   ./longjobs.sh > longjobs.out 2>&1 &
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/userhome/code/MinkLoc3D-master
. ~/.bashrc


#################################################################Ablation Start Part#########################################################################################


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240107_1705_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240107_1705.log
# Avg. top 1% recall: 72.62   Avg. similarity: 595.3726   Avg. recall @N:
# [70.23809524 72.61904762 76.19047619 77.38095238 79.76190476 79.76190476
#  82.14285714 82.14285714 83.33333333 84.52380952 85.71428571 86.9047619
#  88.0952381  89.28571429 89.28571429 89.28571429 90.47619048 90.47619048
#  90.47619048 94.04761905 95.23809524 95.23809524 95.23809524 95.23809524
#  95.23809524]

# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240107_1506_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240107_1506.log
# Avg. top 1% recall: 75.00   Avg. similarity: 585.0261   Avg. recall @N:
# [72.61904762 75.         77.38095238 77.38095238 79.76190476 82.14285714
#  85.71428571 86.9047619  88.0952381  89.28571429 89.28571429 89.28571429
#  89.28571429 90.47619048 90.47619048 91.66666667 92.85714286 94.04761905
#  94.04761905 94.04761905 94.04761905 94.04761905 94.04761905 95.23809524
#  96.42857143]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231227_1908_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231227_1908.log
# Avg. top 1% recall: 82.14   Avg. similarity: 605.2423   Avg. recall @N:
# [76.19047619 82.14285714 84.52380952 85.71428571 86.9047619  86.9047619
#  86.9047619  86.9047619  86.9047619  89.28571429 89.28571429 89.28571429
#  90.47619048 90.47619048 90.47619048 90.47619048 90.47619048 90.47619048
#  90.47619048 90.47619048 91.66666667 92.85714286 92.85714286 92.85714286
#  92.85714286]

#/opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231227_0909_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231227_0909.log
# Avg. top 1% recall: 82.14   Avg. similarity: 620.4941   Avg. recall @N:
# [78.57142857 82.14285714 83.33333333 83.33333333 86.9047619  86.9047619
#  86.9047619  86.9047619  88.0952381  89.28571429 90.47619048 90.47619048
#  90.47619048 90.47619048 90.47619048 90.47619048 90.47619048 90.47619048
#  90.47619048 90.47619048 90.47619048 90.47619048 90.47619048 91.66666667
#  91.66666667]

##nowave
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/basemultimodal.txt --weights ./weights/model_BaseMinkLocMultimodal_20240105_0848_final.pth >./logs/kitti/model_BaseMinkLocMultimodal_20240105_0848.log
# Avg. top 1% recall: 82.14   Avg. similarity: 588.3083   Avg. recall @N:
# [80.95238095 82.14285714 85.71428571 85.71428571 88.0952381  89.28571429
#  89.28571429 89.28571429 89.28571429 89.28571429 90.47619048 90.47619048
#  90.47619048 90.47619048 90.47619048 90.47619048 90.47619048 90.47619048
#  90.47619048 90.47619048 90.47619048 90.47619048 90.47619048 91.66666667
#  92.85714286]

#no basicblock
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240105_1114_final.pth > ./logs/kitti/model_WaveMultimodalInferAll256_20240105_1114.log
# Avg. top 1% recall: 78.57   Avg. similarity: 599.8181   Avg. recall @N:
# [77.38095238 78.57142857 78.57142857 80.95238095 83.33333333 83.33333333
#  84.52380952 85.71428571 85.71428571 85.71428571 88.0952381  88.0952381
#  89.28571429 90.47619048 91.66666667 91.66666667 91.66666667 92.85714286
#  94.04761905 94.04761905 95.23809524 95.23809524 95.23809524 96.42857143
#  96.42857143]



##no wave +add fuse
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/minklocmultimodalAdd.txt --weights ./weights/model_MinkLocMultimodalAdd_20240105_1518_final.pth > ./logs/kitti/model_MinkLocMultimodalAdd_20240105_1518_final.log
# Avg. top 1% recall: 78.57   Avg. similarity: 96.9962   Avg. recall @N:
# [72.61904762 78.57142857 80.95238095 83.33333333 85.71428571 86.9047619
#  88.0952381  89.28571429 90.47619048 90.47619048 91.66666667 91.66666667
#  91.66666667 94.04761905 94.04761905 95.23809524 95.23809524 95.23809524
#  95.23809524 96.42857143 96.42857143 96.42857143 96.42857143 96.42857143
#  96.42857143]


####fuse methods
#/opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodaladd.txt --weights ./weights/model_WaveMultimodalAdd_20231226_0916_final.pth >./logs/kitti/model_WaveMultimodalAdd_20231226_0916.log
# Avg. top 1% recall: 77.38   Avg. similarity: 98.1595   Avg. recall @N:
# [72.61904762 77.38095238 79.76190476 82.14285714 83.33333333 83.33333333
#  84.52380952 84.52380952 84.52380952 84.52380952 84.52380952 85.71428571
#  85.71428571 86.9047619  86.9047619  86.9047619  86.9047619  88.0952381
#  88.0952381  89.28571429 89.28571429 89.28571429 90.47619048 91.66666667
#  91.66666667]

##no svt branches in concate:
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal.txt --weights ./weights/model_WaveMultimodal_20231201_1443_final.pth >./logs/kitti/model_WaveMultimodal_20231201_1443.log
# Avg. top 1% recall: 75.00   Avg. similarity: 61.9053   Avg. recall @N:
# [75.         75.         77.38095238 77.38095238 78.57142857 79.76190476
#  82.14285714 83.33333333 83.33333333 83.33333333 85.71428571 85.71428571
#  86.9047619  86.9047619  86.9047619  86.9047619  86.9047619  89.28571429
#  90.47619048 90.47619048 91.66666667 91.66666667 91.66666667 91.66666667
#  91.66666667]


## different losses
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal_contrast.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231226_0942_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231226_0942.log
# Avg. top 1% recall: 55.95   Avg. similarity: 300.6944   Avg. recall @N:
# [44.04761905 55.95238095 67.85714286 76.19047619 77.38095238 79.76190476
#  79.76190476 80.95238095 80.95238095 80.95238095 80.95238095 80.95238095
#  83.33333333 83.33333333 84.52380952 84.52380952 85.71428571 86.9047619
#  86.9047619  86.9047619  86.9047619  86.9047619  86.9047619  88.0952381
#  90.47619048]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal_ap.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231226_2241_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231226_2241.log
# Avg. top 1% recall: 25.00   Avg. similarity: 294102.7188   Avg. recall @N:
# [20.23809524 25.         29.76190476 35.71428571 38.0952381  42.85714286
#  47.61904762 53.57142857 55.95238095 57.14285714 58.33333333 61.9047619
#  66.66666667 70.23809524 70.23809524 71.42857143 73.80952381 75.
#  76.19047619 77.38095238 77.38095238 77.38095238 77.38095238 78.57142857
#  78.57142857]

###blocks
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_botttle.txt --weights ./weights/model_WaveMultimodalInferAll256_20231226_1809_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231226_1809.log
# Avg. top 1% recall: 73.81   Avg. similarity: 582.7834   Avg. recall @N:
# [71.42857143 73.80952381 75.         76.19047619 77.38095238 80.95238095
#  82.14285714 84.52380952 85.71428571 86.9047619  88.0952381  89.28571429
#  89.28571429 91.66666667 92.85714286 92.85714286 92.85714286 95.23809524
#  95.23809524 95.23809524 95.23809524 95.23809524 95.23809524 95.23809524
#  95.23809524]

# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_ECA.txt --weights ./weights/model_WaveMultimodalInferAll256_20231226_0935_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231226_0935.log
# Avg. top 1% recall: 79.76   Avg. similarity: 594.3126   Avg. recall @N:
# [73.80952381 79.76190476 80.95238095 84.52380952 86.9047619  89.28571429
#  89.28571429 89.28571429 89.28571429 89.28571429 89.28571429 90.47619048
#  91.66666667 92.85714286 92.85714286 94.04761905 94.04761905 94.04761905
#  94.04761905 94.04761905 94.04761905 94.04761905 95.23809524 95.23809524
#  95.23809524]

####nets
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net1.txt --weights ./weights/model_WaveMultimodalInferAll256_20231226_0955_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231226_0955.log
# Avg. top 1% recall: 72.62   Avg. similarity: 623.6335   Avg. recall @N:
# [71.42857143 72.61904762 72.61904762 72.61904762 73.80952381 75.
#  76.19047619 78.57142857 79.76190476 82.14285714 84.52380952 84.52380952
#  86.9047619  88.0952381  88.0952381  89.28571429 89.28571429 89.28571429
#  89.28571429 89.28571429 89.28571429 89.28571429 90.47619048 91.66666667
#  91.66666667]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net2.txt --weights ./weights/model_WaveMultimodalInferAll256_20231226_1447_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231226_1447.log
# Avg. top 1% recall: 72.62   Avg. similarity: 596.9752   Avg. recall @N:
# [64.28571429 72.61904762 77.38095238 78.57142857 79.76190476 82.14285714
#  83.33333333 85.71428571 86.9047619  89.28571429 89.28571429 89.28571429
#  89.28571429 89.28571429 89.28571429 89.28571429 89.28571429 89.28571429
#  92.85714286 94.04761905 95.23809524 95.23809524 95.23809524 96.42857143
#  96.42857143]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net3.txt --weights ./weights/model_WaveMultimodalInferAll256_20231227_0057_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231227_0057.log
# Avg. top 1% recall: 77.38   Avg. similarity: 575.3519   Avg. recall @N:
# [73.80952381 77.38095238 77.38095238 83.33333333 83.33333333 83.33333333
#  83.33333333 85.71428571 85.71428571 85.71428571 86.9047619  86.9047619
#  88.0952381  88.0952381  89.28571429 90.47619048 90.47619048 90.47619048
#  90.47619048 90.47619048 90.47619048 91.66666667 91.66666667 91.66666667
#  91.66666667]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net4.txt --weights ./weights/model_WaveMultimodalInferAll256_20231227_0808_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231227_0808.log
# Avg. top 1% recall: 76.19   Avg. similarity: 578.1385   Avg. recall @N:
# [75.         76.19047619 77.38095238 77.38095238 78.57142857 78.57142857
#  80.95238095 84.52380952 84.52380952 86.9047619  88.0952381  89.28571429
#  89.28571429 90.47619048 91.66666667 92.85714286 94.04761905 94.04761905
#  94.04761905 95.23809524 95.23809524 96.42857143 96.42857143 96.42857143
#  97.61904762]

# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net5.txt --weights ./weights/model_WaveMultimodalInferAll256_20231227_1515_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231227_1515.log
# Avg. top 1% recall: 77.38   Avg. similarity: 631.9747   Avg. recall @N:
# [71.42857143 77.38095238 79.76190476 79.76190476 82.14285714 82.14285714
#  84.52380952 85.71428571 88.0952381  88.0952381  90.47619048 90.47619048
#  90.47619048 92.85714286 92.85714286 94.04761905 94.04761905 94.04761905
#  95.23809524 95.23809524 96.42857143 96.42857143 96.42857143 96.42857143
#  96.42857143]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_net6.txt --weights ./weights/model_WaveMultimodalInferAll256_20231227_2223_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231227_2223.log
# Avg. top 1% recall: 80.95   Avg. similarity: 596.6700   Avg. recall @N:
# [76.19047619 80.95238095 82.14285714 84.52380952 86.9047619  88.0952381
#  89.28571429 91.66666667 92.85714286 92.85714286 94.04761905 94.04761905
#  94.04761905 94.04761905 95.23809524 95.23809524 95.23809524 95.23809524
#  95.23809524 96.42857143 96.42857143 96.42857143 96.42857143 96.42857143
#  96.42857143]

# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_minkloc3dv2net.txt --weights ./weights/model_WaveMultimodalInferAll256_20231228_0532_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231228_0532.log
# Avg. top 1% recall: 82.14   Avg. similarity: 583.8384   Avg. recall @N:
# [76.19047619 82.14285714 85.71428571 88.0952381  88.0952381  88.0952381
#  88.0952381  89.28571429 89.28571429 89.28571429 90.47619048 90.47619048
#  90.47619048 90.47619048 91.66666667 92.85714286 92.85714286 92.85714286
#  92.85714286 95.23809524 95.23809524 95.23809524 95.23809524 96.42857143
#  96.42857143]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_minkloc3dv2net1.txt --weights ./weights/model_WaveMultimodalInferAll256_20231228_1214_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231228_1214.log
# Avg. top 1% recall: 79.76   Avg. similarity: 583.0076   Avg. recall @N:
# [77.38095238 79.76190476 80.95238095 82.14285714 84.52380952 85.71428571
#  86.9047619  90.47619048 91.66666667 91.66666667 91.66666667 91.66666667
#  92.85714286 92.85714286 92.85714286 94.04761905 95.23809524 95.23809524
#  95.23809524 95.23809524 95.23809524 96.42857143 96.42857143 96.42857143
#  96.42857143]

# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal_minkloc3dv2net2.txt --weights ./weights/model_WaveMultimodalInferAll256_20231228_1728_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231228_1728.log
# Avg. top 1% recall: 75.00   Avg. similarity: 583.1105   Avg. recall @N:
# [71.42857143 75.         75.         75.         76.19047619 77.38095238
#  77.38095238 77.38095238 77.38095238 78.57142857 79.76190476 79.76190476
#  82.14285714 84.52380952 84.52380952 84.52380952 85.71428571 86.9047619
#  88.0952381  88.0952381  89.28571429 89.28571429 89.28571429 89.28571429
#  89.28571429]

##rgb resnet
##no svt branches:
#/opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergb256.txt --weights ./weights/model_WaveRGB256_20231229_0904_final.pth >./logs/kitti/model_WaveRGB256_20231229_0904.log
# Avg. top 1% recall: 71.43   Avg. similarity: 25.8160   Avg. recall @N:
# [69.04761905 71.42857143 73.80952381 73.80952381 76.19047619 77.38095238
#  77.38095238 79.76190476 82.14285714 82.14285714 82.14285714 82.14285714
#  82.14285714 82.14285714 82.14285714 82.14285714 82.14285714 82.14285714
#  83.33333333 84.52380952 85.71428571 85.71428571 85.71428571 85.71428571
#  85.71428571]

# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinfer_rgb34.txt --weights ./weights/model_WaveMultimodalInfer_rgb34_20231227_0845_final.pth >./logs/kitti/model_WaveMultimodalInfer_rgb34_20231227_0845.log
# Avg. top 1% recall: 78.57   Avg. similarity: 574.5363   Avg. recall @N:
# [73.80952381 78.57142857 79.76190476 79.76190476 79.76190476 82.14285714
#  82.14285714 83.33333333 85.71428571 86.9047619  86.9047619  86.9047619
#  88.0952381  88.0952381  89.28571429 89.28571429 89.28571429 89.28571429
#  89.28571429 90.47619048 91.66666667 92.85714286 94.04761905 94.04761905
#  94.04761905]


#/opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergbtest.txt --weights ./weights/model_WaveRGBnetTest_20231228_0944_final.pth >./logs/kitti/model_WaveRGBnetTest_20231228_0944.log
# Avg. top 1% recall: 72.62   Avg. similarity: 22.8848   Avg. recall @N:
# [70.23809524 72.61904762 72.61904762 73.80952381 75.         76.19047619
#  76.19047619 77.38095238 77.38095238 77.38095238 79.76190476 82.14285714
#  82.14285714 82.14285714 83.33333333 84.52380952 84.52380952 84.52380952
#  84.52380952 84.52380952 85.71428571 86.9047619  89.28571429 90.47619048
#  91.66666667]

##weights
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal1000.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231226_1023_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231226_1023.log
# Avg. top 1% recall: 77.38   Avg. similarity: 558.4252   Avg. recall @N:
# [76.19047619 77.38095238 82.14285714 85.71428571 88.0952381  90.47619048
#  90.47619048 90.47619048 90.47619048 90.47619048 91.66666667 91.66666667
#  91.66666667 91.66666667 91.66666667 92.85714286 92.85714286 92.85714286
#  92.85714286 92.85714286 92.85714286 94.04761905 94.04761905 94.04761905
#  94.04761905]

# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal433.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231226_1514_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231226_1514.log
# Avg. top 1% recall: 70.24   Avg. similarity: 601.1916   Avg. recall @N:
# [66.66666667 70.23809524 73.80952381 73.80952381 75.         76.19047619
#  77.38095238 77.38095238 78.57142857 79.76190476 80.95238095 83.33333333
#  85.71428571 85.71428571 86.9047619  86.9047619  88.0952381  88.0952381
#  88.0952381  88.0952381  88.0952381  90.47619048 90.47619048 90.47619048
#  91.66666667]

# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal505.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231227_0428_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231227_0428.log
# Avg. top 1% recall: 79.76   Avg. similarity: 589.4412   Avg. recall @N:
# [76.19047619 79.76190476 82.14285714 82.14285714 83.33333333 83.33333333
#  83.33333333 83.33333333 84.52380952 84.52380952 85.71428571 85.71428571
#  88.0952381  89.28571429 90.47619048 90.47619048 91.66666667 91.66666667
#  92.85714286 92.85714286 92.85714286 92.85714286 92.85714286 92.85714286
#  92.85714286]

# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal514.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231227_1042_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231227_1042.log
# Avg. top 1% recall: 76.19   Avg. similarity: 592.0054   Avg. recall @N:
# [73.80952381 76.19047619 76.19047619 77.38095238 78.57142857 80.95238095
#  82.14285714 82.14285714 82.14285714 84.52380952 86.9047619  86.9047619
#  86.9047619  86.9047619  86.9047619  86.9047619  89.28571429 89.28571429
#  89.28571429 89.28571429 91.66666667 91.66666667 92.85714286 94.04761905
#  94.04761905]
 
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal523.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231227_1747_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231227_1747.log
# Avg. top 1% recall: 76.19   Avg. similarity: 568.1359   Avg. recall @N:
# [75.         76.19047619 76.19047619 76.19047619 77.38095238 79.76190476
#  79.76190476 79.76190476 79.76190476 79.76190476 79.76190476 80.95238095
#  82.14285714 82.14285714 83.33333333 83.33333333 86.9047619  86.9047619
#  86.9047619  88.0952381  88.0952381  89.28571429 89.28571429 90.47619048
#  90.47619048]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal541.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231227_2336_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231227_2336.log
# Avg. top 1% recall: 75.00   Avg. similarity: 577.0303   Avg. recall @N:
# [73.80952381 75.         77.38095238 77.38095238 78.57142857 79.76190476
#  79.76190476 79.76190476 82.14285714 83.33333333 84.52380952 84.52380952
#  85.71428571 86.9047619  86.9047619  88.0952381  88.0952381  89.28571429
#  89.28571429 89.28571429 89.28571429 89.28571429 89.28571429 89.28571429
#  90.47619048]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal532.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231228_0647_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20231228_0647.log
# Avg. top 1% recall: 73.81   Avg. similarity: 598.5370   Avg. recall @N:
# [72.61904762 73.80952381 76.19047619 78.57142857 78.57142857 78.57142857
#  80.95238095 80.95238095 83.33333333 84.52380952 84.52380952 85.71428571
#  85.71428571 85.71428571 85.71428571 85.71428571 85.71428571 85.71428571
#  85.71428571 85.71428571 85.71428571 85.71428571 86.9047619  88.0952381
#  88.0952381 ]

#weights
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal91.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240105_1457_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240105_1457.log
# Avg. top 1% recall: 76.19   Avg. similarity: 569.2877   Avg. recall @N:
# [72.61904762 76.19047619 78.57142857 83.33333333 84.52380952 86.9047619
#  86.9047619  89.28571429 90.47619048 91.66666667 92.85714286 92.85714286
#  92.85714286 94.04761905 94.04761905 94.04761905 94.04761905 94.04761905
#  94.04761905 95.23809524 95.23809524 95.23809524 95.23809524 95.23809524
#  95.23809524]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal82.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240105_1947_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240105_1947.log
# Avg. top 1% recall: 73.81   Avg. similarity: 579.3084   Avg. recall @N:
# [70.23809524 73.80952381 73.80952381 73.80952381 75.         76.19047619
#  80.95238095 80.95238095 83.33333333 86.9047619  88.0952381  89.28571429
#  89.28571429 90.47619048 91.66666667 91.66666667 91.66666667 94.04761905
#  95.23809524 95.23809524 95.23809524 95.23809524 95.23809524 95.23809524
#  95.23809524]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal73.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240106_0036_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240106_0036.log
# Avg. top 1% recall: 73.81   Avg. similarity: 564.8259   Avg. recall @N:
# [72.61904762 73.80952381 78.57142857 82.14285714 83.33333333 85.71428571
#  85.71428571 89.28571429 89.28571429 89.28571429 89.28571429 90.47619048
#  90.47619048 91.66666667 91.66666667 91.66666667 91.66666667 91.66666667
#  91.66666667 91.66666667 91.66666667 91.66666667 91.66666667 92.85714286
#  95.23809524]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal64.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240106_0525_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240106_0525.log
# Avg. top 1% recall: 78.57   Avg. similarity: 590.6501   Avg. recall @N:
# [70.23809524 78.57142857 78.57142857 79.76190476 80.95238095 83.33333333
#  88.0952381  89.28571429 89.28571429 89.28571429 89.28571429 90.47619048
#  91.66666667 91.66666667 91.66666667 91.66666667 92.85714286 92.85714286
#  94.04761905 94.04761905 94.04761905 94.04761905 94.04761905 94.04761905
#  95.23809524]
 
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal46.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240106_1012_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240106_1012.log
# Avg. top 1% recall: 71.43   Avg. similarity: 619.4876   Avg. recall @N:
# [64.28571429 71.42857143 73.80952381 78.57142857 79.76190476 79.76190476
#  80.95238095 82.14285714 82.14285714 82.14285714 82.14285714 82.14285714
#  84.52380952 85.71428571 85.71428571 85.71428571 85.71428571 89.28571429
#  89.28571429 90.47619048 90.47619048 91.66666667 91.66666667 91.66666667
#  91.66666667]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal37.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240106_1446_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240106_1446.log
# Avg. top 1% recall: 80.95   Avg. similarity: 605.3303   Avg. recall @N:
# [75.         80.95238095 83.33333333 85.71428571 86.9047619  86.9047619
#  86.9047619  89.28571429 92.85714286 92.85714286 92.85714286 94.04761905
#  94.04761905 94.04761905 94.04761905 94.04761905 94.04761905 94.04761905
#  95.23809524 95.23809524 95.23809524 95.23809524 95.23809524 95.23809524
#  96.42857143]

# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal37.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240107_1024_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240107_1024.log
# Avg. top 1% recall: 76.19   Avg. similarity: 622.7543   Avg. recall @N:
# [71.42857143 76.19047619 77.38095238 79.76190476 79.76190476 79.76190476
#  79.76190476 79.76190476 79.76190476 79.76190476 79.76190476 79.76190476
#  79.76190476 82.14285714 82.14285714 84.52380952 84.52380952 84.52380952
#  84.52380952 84.52380952 84.52380952 84.52380952 84.52380952 85.71428571
#  86.9047619 ]
 
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal37.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240107_1228_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240107_1228.log
# Avg. top 1% recall: 73.81   Avg. similarity: 581.2362   Avg. recall @N:
# [70.23809524 73.80952381 75.         76.19047619 78.57142857 78.57142857
#  80.95238095 83.33333333 84.52380952 88.0952381  89.28571429 91.66666667
#  91.66666667 91.66666667 91.66666667 92.85714286 92.85714286 92.85714286
#  92.85714286 92.85714286 94.04761905 94.04761905 95.23809524 95.23809524
#  95.23809524]



# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal28.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240106_1933_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240106_1933.log
# Avg. top 1% recall: 73.81   Avg. similarity: 601.6595   Avg. recall @N:
# [70.23809524 73.80952381 73.80952381 76.19047619 77.38095238 77.38095238
#  79.76190476 80.95238095 82.14285714 82.14285714 82.14285714 82.14285714
#  84.52380952 84.52380952 85.71428571 85.71428571 85.71428571 85.71428571
#  85.71428571 85.71428571 86.9047619  88.0952381  89.28571429 89.28571429
#  90.47619048]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal19.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240107_0012_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240107_0012.log
# Avg. top 1% recall: 75.00   Avg. similarity: 474.7071   Avg. recall @N:
# [60.71428571 75.         76.19047619 78.57142857 82.14285714 82.14285714
#  82.14285714 82.14285714 83.33333333 83.33333333 85.71428571 85.71428571
#  86.9047619  86.9047619  86.9047619  88.0952381  88.0952381  88.0952381
#  90.47619048 91.66666667 91.66666667 91.66666667 91.66666667 91.66666667
#  92.85714286]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal111.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240107_1450_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240107_1450.log
# Avg. top 1% recall: 78.57   Avg. similarity: 638.7830   Avg. recall @N:
# [72.61904762 78.57142857 82.14285714 84.52380952 84.52380952 84.52380952
#  84.52380952 84.52380952 86.9047619  88.0952381  88.0952381  88.0952381
#  88.0952381  88.0952381  88.0952381  88.0952381  88.0952381  88.0952381
#  89.28571429 89.28571429 89.28571429 89.28571429 89.28571429 89.28571429
#  90.47619048]
 
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal110.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240107_1949_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240107_1949.log
# Avg. top 1% recall: 78.57   Avg. similarity: 607.4027   Avg. recall @N:
# [72.61904762 78.57142857 78.57142857 80.95238095 82.14285714 83.33333333
#  85.71428571 85.71428571 85.71428571 88.0952381  88.0952381  88.0952381
#  89.28571429 89.28571429 90.47619048 90.47619048 91.66666667 91.66666667
#  91.66666667 91.66666667 91.66666667 91.66666667 91.66666667 91.66666667
#  91.66666667]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal119.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240108_0043_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240108_0043.log
# Avg. top 1% recall: 75.00   Avg. similarity: 602.6703   Avg. recall @N:
# [72.61904762 75.         79.76190476 80.95238095 80.95238095 80.95238095
#  80.95238095 80.95238095 83.33333333 83.33333333 83.33333333 83.33333333
#  83.33333333 85.71428571 85.71428571 85.71428571 85.71428571 86.9047619
#  88.0952381  89.28571429 90.47619048 90.47619048 90.47619048 90.47619048
#  90.47619048]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal901.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240108_1016_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240108_1016.log
# Avg. top 1% recall: 72.62   Avg. similarity: 569.1006   Avg. recall @N:
# [71.42857143 72.61904762 75.         76.19047619 77.38095238 78.57142857
#  79.76190476 79.76190476 82.14285714 83.33333333 83.33333333 83.33333333
#  83.33333333 83.33333333 83.33333333 85.71428571 85.71428571 86.9047619
#  88.0952381  88.0952381  88.0952381  89.28571429 89.28571429 90.47619048
#  91.66666667]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal802.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240108_1438_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240108_1438.log
# Avg. top 1% recall: 77.38   Avg. similarity: 572.5930   Avg. recall @N:
# [72.61904762 77.38095238 82.14285714 84.52380952 84.52380952 85.71428571
#  85.71428571 85.71428571 85.71428571 85.71428571 85.71428571 88.0952381
#  88.0952381  89.28571429 89.28571429 90.47619048 90.47619048 90.47619048
#  91.66666667 91.66666667 91.66666667 91.66666667 91.66666667 91.66666667
#  91.66666667]

# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal119.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240108_0043_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240108_0043.log
# Avg. top 1% recall: 75.00   Avg. similarity: 602.6704   Avg. recall @N:
# [72.61904762 75.         79.76190476 80.95238095 80.95238095 80.95238095
#  80.95238095 80.95238095 83.33333333 83.33333333 83.33333333 83.33333333
#  83.33333333 85.71428571 85.71428571 85.71428571 85.71428571 86.9047619
#  88.0952381  89.28571429 90.47619048 90.47619048 90.47619048 90.47619048
#  90.47619048]
 
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal110.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240107_1949_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240107_1949.log
# Avg. top 1% recall: 78.57   Avg. similarity: 607.4027   Avg. recall @N:
# [72.61904762 78.57142857 78.57142857 80.95238095 82.14285714 83.33333333
#  85.71428571 85.71428571 85.71428571 88.0952381  88.0952381  88.0952381
#  89.28571429 89.28571429 90.47619048 90.47619048 91.66666667 91.66666667
#  91.66666667 91.66666667 91.66666667 91.66666667 91.66666667 91.66666667
#  91.66666667]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal114.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240108_0955_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240108_0955.log
# Avg. top 1% recall: 77.38   Avg. similarity: 604.1086   Avg. recall @N:
# [75.         77.38095238 79.76190476 82.14285714 82.14285714 83.33333333
#  83.33333333 85.71428571 86.9047619  86.9047619  86.9047619  86.9047619
#  88.0952381  89.28571429 89.28571429 90.47619048 90.47619048 92.85714286
#  94.04761905 94.04761905 94.04761905 94.04761905 94.04761905 94.04761905
#  94.04761905]

# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal114.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240108_1440_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240108_1440.log
# Avg. top 1% recall: 73.81   Avg. similarity: 598.6882   Avg. recall @N:
# [71.42857143 73.80952381 76.19047619 76.19047619 77.38095238 79.76190476
#  80.95238095 80.95238095 82.14285714 82.14285714 83.33333333 85.71428571
#  86.9047619  88.0952381  89.28571429 90.47619048 90.47619048 90.47619048
#  91.66666667 91.66666667 92.85714286 92.85714286 92.85714286 92.85714286
#  94.04761905]


#/opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal442.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20240108_0955_final.pth >./logs/kitti/model_WaveMultimodalInferAll256_20240108_0955.log
# Avg. top 1% recall: 77.38   Avg. similarity: 604.1086   Avg. recall @N:
# [75.         77.38095238 79.76190476 82.14285714 82.14285714 83.33333333
#  83.33333333 85.71428571 86.9047619  86.9047619  86.9047619  86.9047619
#  88.0952381  89.28571429 89.28571429 90.47619048 90.47619048 92.85714286
#  94.04761905 94.04761905 94.04761905 94.04761905 94.04761905 94.04761905
#  94.04761905]


#################################################################Ablation End Part#########################################################################################


#################################################################SVT Start Part#########################################################################################
###SVT for point clouds 
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svt.txt --weights ./weights/model_SVT_20231223_1539_final.pth >./logs/kitti/model_SVT_20231223_1539_final.log
# Avg. top 1% recall: 82.14   Avg. similarity: 44.2822   Avg. recall @N:
# [76.19047619 82.14285714 84.52380952 84.52380952 84.52380952 86.9047619
#  86.9047619  86.9047619  86.9047619  86.9047619  90.47619048 90.47619048
#  90.47619048 90.47619048 90.47619048 90.47619048 91.66666667 92.85714286
#  92.85714286 92.85714286 92.85714286 92.85714286 92.85714286 94.04761905
#  94.04761905]
 
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtqi.txt --weights ./weights/model_SVTQI_20231223_1551_final.pth >./logs/kitti/model_SVTQI_20231223_1551.log
# Avg. top 1% recall: 77.38   Avg. similarity: 60.2841   Avg. recall @N:
# [73.80952381 77.38095238 79.76190476 79.76190476 80.95238095 80.95238095
#  80.95238095 83.33333333 84.52380952 84.52380952 85.71428571 85.71428571
#  85.71428571 85.71428571 86.9047619  86.9047619  88.0952381  88.0952381
#  88.0952381  88.0952381  88.0952381  88.0952381  88.0952381  88.0952381
#  88.0952381 ]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtinferall128.txt --weights ./weights/model_SVTinferall128_20231224_0021_final.pth >./logs/kitti/model_SVTinferall128_20231224_0021.log
# Avg. top 1% recall: 75.00   Avg. similarity: 469.9753   Avg. recall @N:
# [72.61904762 75.         82.14285714 84.52380952 85.71428571 86.9047619
#  86.9047619  89.28571429 89.28571429 89.28571429 89.28571429 89.28571429
#  90.47619048 90.47619048 90.47619048 90.47619048 90.47619048 90.47619048
#  90.47619048 91.66666667 91.66666667 91.66666667 91.66666667 92.85714286
#  92.85714286]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtqiinferall128.txt --weights ./weights/model_SVTQIinferall128_20231224_0029_final.pth >./logs/kitti/model_SVTQIinferall128_20231224_0029.log
# Loading weights: ./weights/model_SVTQIinferall128_20231224_0029_final.pth
# Dataset: KITTI
# Avg. top 1% recall: 73.81   Avg. similarity: 331.2945   Avg. recall @N:
# [72.61904762 73.80952381 77.38095238 79.76190476 80.95238095 82.14285714
#  84.52380952 84.52380952 85.71428571 85.71428571 85.71428571 86.9047619
#  89.28571429 89.28571429 89.28571429 89.28571429 90.47619048 90.47619048
#  91.66666667 91.66666667 92.85714286 92.85714286 92.85714286 92.85714286
#  92.85714286]
 

# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtinferall256.txt --weights ./weights/model_SVTinferall256_20231224_0900_final.pth >./logs/kitti/model_SVTinferall256_20231224_0900.log
# Avg. top 1% recall: 83.33   Avg. similarity: 694.9637   Avg. recall @N:
# [79.76190476 83.33333333 85.71428571 85.71428571 88.0952381  88.0952381
#  88.0952381  88.0952381  88.0952381  88.0952381  88.0952381  88.0952381
#  88.0952381  88.0952381  89.28571429 89.28571429 89.28571429 89.28571429
#  90.47619048 91.66666667 91.66666667 92.85714286 92.85714286 92.85714286
#  92.85714286]


# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/svtqiinferall256.txt --weights ./weights/model_SVTQIinferall256_20231224_0920_final.pth >./logs/kitti/model_SVTQIinferall256_20231224_0920.log
# Avg. top 1% recall: 80.95   Avg. similarity: 527.2551   Avg. recall @N:
# [77.38095238 80.95238095 80.95238095 84.52380952 88.0952381  89.28571429
#  89.28571429 90.47619048 91.66666667 91.66666667 94.04761905 94.04761905
#  95.23809524 95.23809524 95.23809524 95.23809524 95.23809524 95.23809524
#  95.23809524 97.61904762 97.61904762 97.61904762 97.61904762 97.61904762
#  97.61904762]
#################################################################SVT End Part#########################################################################################



##############################################################Point CLouds & RGB Start ################################################################################

# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal352.txt --model_config ./models/wavemultimodalinferall.txt --weights ./weights/model_WaveMultimodalInferAll_20231226_0835_final.pth >./logs/kitti/model_WaveMultimodalInferAll_20231226_0835.log
# Avg. top 1% recall: 79.76   Avg. similarity: 416.6180   Avg. recall @N:
# [75.         79.76190476 80.95238095 80.95238095 85.71428571 85.71428571
#  85.71428571 88.0952381  88.0952381  89.28571429 89.28571429 91.66666667
#  91.66666667 92.85714286 92.85714286 94.04761905 94.04761905 94.04761905
#  94.04761905 94.04761905 94.04761905 94.04761905 94.04761905 94.04761905
#  94.04761905]


#/opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall.txt --weights ./weights/model_WaveMultimodalInferAll_20231222_2043_final.pth >./logs/kitti/model_WaveMultimodalInferAll_20231222_2043_final.log
# Avg. top 1% recall: 75.00   Avg. similarity: 376.1099   Avg. recall @N:
# [69.04761905 75.         80.95238095 84.52380952 85.71428571 85.71428571
#  86.9047619  91.66666667 91.66666667 94.04761905 94.04761905 94.04761905
#  95.23809524 95.23809524 95.23809524 95.23809524 95.23809524 95.23809524
#  95.23809524 95.23809524 95.23809524 95.23809524 95.23809524 95.23809524
#  95.23809524]

##no SVT series branches can work well
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231223_0949_final.pth >./logs/kitti/wavemultimodalinferall256.log
# Avg. top 1% recall: 82.14   Avg. similarity: 595.0306   Avg. recall @N:
# [77.38095238 82.14285714 85.71428571 85.71428571 85.71428571 85.71428571
#  85.71428571 86.9047619  88.0952381  90.47619048 94.04761905 94.04761905
#  94.04761905 94.04761905 94.04761905 94.04761905 94.04761905 94.04761905
#  94.04761905 94.04761905 94.04761905 94.04761905 94.04761905 94.04761905
#  94.04761905]

# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal352.txt --model_config ./models/wavemultimodalinfer.txt --weights ./weights/model_WaveMultimodalInfer_20231225_1018_final.pth >./logs/kitti/model_WaveMultimodalInfer_20231225_1018.log
# Avg. top 1% recall: 77.38   Avg. similarity: 60.5469   Avg. recall @N:
# [75.         77.38095238 77.38095238 79.76190476 80.95238095 82.14285714
#  83.33333333 84.52380952 84.52380952 84.52380952 84.52380952 84.52380952
#  85.71428571 85.71428571 86.9047619  86.9047619  86.9047619  86.9047619
#  88.0952381  89.28571429 89.28571429 90.47619048 90.47619048 90.47619048
#  90.47619048]

##SVT series branches added
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal352.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231224_1033_final.pth >./logs/kitti/wavemultimodalinferall256_352.log
# Dataset: KITTI
# Avg. top 1% recall: 82.14   Avg. similarity: 616.2609   Avg. recall @N:
# [73.80952381 82.14285714 84.52380952 86.9047619  86.9047619  88.0952381
#  88.0952381  89.28571429 89.28571429 89.28571429 89.28571429 89.28571429
#  89.28571429 90.47619048 90.47619048 90.47619048 90.47619048 91.66666667
#  91.66666667 91.66666667 91.66666667 92.85714286 94.04761905 94.04761905
#  94.04761905]
 
##SVT series branches added
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal256infer.txt --weights ./weights/model_WaveMultimodalInfer256_20231224_1031_final.pth >./logs/kitti/wavemultimodalinfer256_again.log
# Dataset: KITTI
# Avg. top 1% recall: 71.43   Avg. similarity: 75.2675   Avg. recall @N:
# [69.04761905 71.42857143 75.         78.57142857 78.57142857 80.95238095
#  83.33333333 83.33333333 83.33333333 84.52380952 84.52380952 84.52380952
#  85.71428571 85.71428571 85.71428571 85.71428571 85.71428571 85.71428571
#  86.9047619  86.9047619  88.0952381  88.0952381  88.0952381  88.0952381
#  88.0952381 ]


##SVT series branches added
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal352.txt --model_config ./models/wavemultimodal256infer.txt --weights ./weights/model_WaveMultimodalInfer256_20231224_1531_final.pth >./logs/kitti/model_WaveMultimodalInfer256_20231224_1531_final.log
# Avg. top 1% recall: 77.38   Avg. similarity: 83.6625   Avg. recall @N:
# [71.42857143 77.38095238 84.52380952 84.52380952 85.71428571 85.71428571
#  88.0952381  88.0952381  88.0952381  89.28571429 91.66666667 91.66666667
#  91.66666667 92.85714286 92.85714286 94.04761905 94.04761905 94.04761905
#  94.04761905 94.04761905 94.04761905 94.04761905 95.23809524 95.23809524
#  95.23809524]

#python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal.txt --weights ./weights/model_WaveMultimodal_20231128_1128_final.pth

##PCS+RGB infer 256D
# python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodal256infer.txt --weights ./weights/model_WaveMultimodalInfer256_20231206_2307_final.pth >kitti_ours_all256.log
# Dataset: KITTI
# Avg. top 1% recall: 75.00   Avg. similarity: 77.1971   Avg. recall @N:
# [67.85714286 75.         78.57142857 82.14285714 83.33333333 85.71428571
#  89.28571429 90.47619048 90.47619048 90.47619048 91.66666667 92.85714286
#  92.85714286 94.04761905 94.04761905 94.04761905 95.23809524 95.23809524
#  95.23809524 95.23809524 95.23809524 95.23809524 95.23809524 95.23809524
#  95.23809524]


##PCS+RGB infer 128D
#python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinfer.txt --weights ./weights/model_WaveMultimodalInfer_20231202_1229_final.pth
# Avg. top 1% recall: 75.00   Avg. similarity: 110.7516   Avg. recall @N:
# [73.80952381 75.         79.76190476 80.95238095 84.52380952 85.71428571
#  85.71428571 85.71428571 85.71428571 86.9047619  89.28571429 89.28571429
#  90.47619048 90.47619048 90.47619048 90.47619048 90.47619048 90.47619048
#  90.47619048 90.47619048 90.47619048 91.66666667 91.66666667 92.85714286
#  92.85714286]
 
#mink_quantization_size = [2.5, 2.0, 0.42]
# Avg. top 1% recall: 78.57   Avg. similarity: 88.3528   Avg. recall @N:
# [72.61904762 78.57142857 79.76190476 80.95238095 83.33333333 83.33333333
#  84.52380952 86.9047619  88.0952381  88.0952381  89.28571429 89.28571429
#  91.66666667 91.66666667 91.66666667 91.66666667 91.66666667 91.66666667
#  91.66666667 92.85714286 94.04761905 94.04761905 95.23809524 95.23809524
#  95.23809524]

#python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/wavemultimodalinfer.txt --weights ./weights/model_WaveMultimodal_20231201_1443_final.pth
#mink_quantization_size = [2.5, 2.0, 0.42]
# Avg. top 1% recall: 76.19   Avg. similarity: 137.5359   Avg. recall @N:
# [72.61904762 76.19047619 79.76190476 80.95238095 84.52380952 84.52380952
#  84.52380952 85.71428571 85.71428571 85.71428571 86.9047619  86.9047619
#  86.9047619  88.0952381  88.0952381  90.47619048 90.47619048 90.47619048
#  90.47619048 92.85714286 92.85714286 94.04761905 94.04761905 94.04761905
#  94.04761905]

##############################################################Point CLouds & RGB End ################################################################################



##############################################################Point CLouds & RGB Sketches Start ################################################################################
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/wavemultimodalinfer.txt --weights ./weights/model_WaveMultimodalInfer_20231225_2005_final.pth >./logs/kitti/model_WaveMultimodalInfer_20231225_2005_final.log
# Avg. top 1% recall: 77.38   Avg. similarity: 47.9128   Avg. recall @N:
# [69.04761905 77.38095238 80.95238095 83.33333333 83.33333333 84.52380952
#  86.9047619  88.0952381  89.28571429 89.28571429 90.47619048 91.66666667
#  92.85714286 92.85714286 94.04761905 94.04761905 95.23809524 95.23809524
#  95.23809524 95.23809524 95.23809524 95.23809524 95.23809524 95.23809524
#  95.23809524]

##no SVT series branches can work well
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231223_0950_final.pth >./logs/kitti/wavemultimodalinferall256_PCSRGBs.log
# Dataset: KITTI
# Avg. top 1% recall: 80.95   Avg. similarity: 585.5558   Avg. recall @N:
# [78.57142857 80.95238095 84.52380952 84.52380952 84.52380952 84.52380952
#  84.52380952 85.71428571 88.0952381  88.0952381  89.28571429 89.28571429
#  90.47619048 91.66666667 91.66666667 92.85714286 92.85714286 92.85714286
#  92.85714286 92.85714286 92.85714286 92.85714286 92.85714286 92.85714286
#  95.23809524]

# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal_sketchRGB352.txt --model_config ./models/wavemultimodalinfer.txt --weights ./weights/model_WaveMultimodalInfer_20231225_1008_final.pth >./logs/kitti/model_WaveMultimodalInfer_20231225_1008.log
# Avg. top 1% recall: 71.43   Avg. similarity: 62.9968   Avg. recall @N:
# [67.85714286 71.42857143 73.80952381 77.38095238 79.76190476 82.14285714
#  84.52380952 84.52380952 85.71428571 85.71428571 86.9047619  88.0952381
#  89.28571429 90.47619048 90.47619048 90.47619048 90.47619048 91.66666667
#  92.85714286 94.04761905 94.04761905 94.04761905 95.23809524 95.23809524
#  95.23809524]

##SVT series branches added
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal_sketchRGB352.txt --model_config ./models/wavemultimodalinferall256.txt --weights ./weights/model_WaveMultimodalInferAll256_20231223_1601_final.pth >./logs/kitti/wavemultimodalinferall256_PCSRGBs352.log
# Avg. top 1% recall: 80.95   Avg. similarity: 591.6880   Avg. recall @N:
# [73.80952381 80.95238095 82.14285714 83.33333333 85.71428571 88.0952381
#  91.66666667 92.85714286 92.85714286 94.04761905 94.04761905 94.04761905
#  94.04761905 94.04761905 94.04761905 94.04761905 94.04761905 94.04761905
#  94.04761905 94.04761905 94.04761905 94.04761905 94.04761905 94.04761905
#  94.04761905]

##SVT series branches added
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal_sketchRGB.txt --model_config ./models/wavemultimodal256infer.txt --weights ./weights/model_WaveMultimodalInfer256_20231224_1734_final.pth >./logs/kitti/wavemultimodalinfer56_PCSRGBs.log
# Dataset: KITTI
# Avg. top 1% recall: 75.00   Avg. similarity: 77.5371   Avg. recall @N:
# [73.80952381 75.         76.19047619 77.38095238 80.95238095 82.14285714
#  82.14285714 83.33333333 83.33333333 86.9047619  88.0952381  89.28571429
#  89.28571429 90.47619048 90.47619048 90.47619048 91.66666667 91.66666667
#  91.66666667 92.85714286 92.85714286 92.85714286 92.85714286 92.85714286
#  94.04761905]
 
##SVT series branches added
# /opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal_sketchRGB352.txt --model_config ./models/wavemultimodal256infer.txt --weights ./weights/model_WaveMultimodalInfer256_20231224_2346_final.pth >./logs/kitti/wavemultimodalinfer256_PCSRGBs352.log
# Dataset: KITTI
# Avg. top 1% recall: 83.33   Avg. similarity: 84.6553   Avg. recall @N:
# [80.95238095 83.33333333 84.52380952 85.71428571 85.71428571 88.0952381
#  88.0952381  88.0952381  89.28571429 89.28571429 89.28571429 89.28571429
#  89.28571429 89.28571429 89.28571429 89.28571429 89.28571429 89.28571429
#  90.47619048 90.47619048 90.47619048 90.47619048 90.47619048 90.47619048
#  90.47619048]

##############################################################Point CLouds & RGB Sketches End ################################################################################




##############################################################All Sketches  ################################################################################

# python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal_sketch.txt --model_config ./models/wavemultimodalinfer.txt --weights ./weights/model_WaveMultimodalInfer_20231212_1006_final.pth >kitti_ours_all_sketch.log

# python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal_sketch_nonuniform.txt --model_config ./models/wavemultimodalinfer.txt --weights ./weights/model_WaveMultimodalInfer_20231214_0722_final.pth >kitti_ours_all_sketch_uniform.log

# python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal_sketch_nonuniform.txt --model_config ./models/wavemultimodal256infer.txt --weights ./weights/model_WaveMultimodalInfer256_20231214_1128_final.pth >kitti_ours_all_sketch_uniform256.log




############################################################## Point Clouds & its sketches only Start ####################################################################################
###PCS 256D
#python ./eval/evaluate_kitti.py --config ./config/config_baseline.txt --model_config ./models/wave3d256.txt --weights ./weights/model_Wave3D_20231128_1620_final.pth
# Avg. top 1% recall: 64.29   Avg. similarity: 49.6480   Avg. recall @N:
# [60.71428571 64.28571429 66.66666667 66.66666667 66.66666667 70.23809524
#  72.61904762 72.61904762 77.38095238 79.76190476 82.14285714 83.33333333
#  84.52380952 84.52380952 84.52380952 84.52380952 84.52380952 84.52380952
#  88.0952381  89.28571429 89.28571429 91.66666667 92.85714286 92.85714286
#  92.85714286]

#mink_quantization_size = [2.5, 2.0, 0.42] 
# #Avg. top 1% recall: 77.38   Avg. similarity: 43.1598   Avg. recall @N:
# [72.61904762 77.38095238 78.57142857 79.76190476 82.14285714 82.14285714
#  82.14285714 83.33333333 84.52380952 85.71428571 86.9047619  86.9047619
#  90.47619048 91.66666667 91.66666667 91.66666667 91.66666667 91.66666667
#  92.85714286 94.04761905 94.04761905 96.42857143 96.42857143 96.42857143
#  96.42857143]

##PCS 128D
# python ./eval/evaluate_kitti.py --config ./config/config_baseline.txt --model_config ./models/wave3d.txt --weights ./weights/model_Wave3D_20231206_1503_final.pth
#mink_quantization_size = [2.5, 2.0, 0.42]
# Dataset: KITTI
# Avg. top 1% recall: 72.62   Avg. similarity: 33.2478   Avg. recall @N:
# [66.66666667 72.61904762 78.57142857 80.95238095 82.14285714 85.71428571
#  86.9047619  86.9047619  86.9047619  86.9047619  89.28571429 89.28571429
#  90.47619048 91.66666667 91.66666667 91.66666667 91.66666667 91.66666667
#  91.66666667 91.66666667 91.66666667 91.66666667 92.85714286 94.04761905
#  94.04761905]

#python ./eval/evaluate_kitti.py --config ./config/config_baseline.txt --model_config ./models/wave3d256.txt --weights ./weights/model_Wave3D_20231128_1620_final.pth >kitti_ours_pc256.log

# python ./eval/evaluate_kitti.py --config ./config/config_pc_graph.txt --model_config ./models/wave3d.txt --weights ./weights/model_Wave3D_20231207_0418_final.pth >kitti_ours_pc_sketch.log

# python ./eval/evaluate_kitti.py --config ./config/config_pc_graph.txt --model_config ./models/wave3d256.txt --weights ./weights/model_Wave3D_20231207_1050_final.pth >kitti_ours_pc_sketch256.log

# python ./eval/evaluate_kitti.py --config ./config/config_pc_nonuniform.txt --model_config ./models/wave3d.txt --weights ./weights/model_Wave3D_20231207_1025_final.pth >kitti_ours_pc_uniform.log

# python ./eval/evaluate_kitti.py --config ./config/config_pc_nonuniform.txt --model_config ./models/wave3d256.txt --weights ./weights/model_Wave3D_20231207_1700_final.pth >kitti_ours_pc_uniform256.log

############################################################## Point Clouds only End ######################################################################################################


 
############################################################## RGS only Start ######################################################################################################

### RGB 128D
# python ./eval/evaluate_kitti.py --config ./config/config_baseline_rgb.txt --model_config ./models/wavergb.txt --weights ./weights/model_WaveRGB_20231204_1649_final.pth
#mink_quantization_size = [2.5, 2.0, 0.42]
# Avg. top 1% recall: 75.00   Avg. similarity: 17.1143   Avg. recall @N:
# [75.         75.         77.38095238 79.76190476 79.76190476 82.14285714
#  82.14285714 82.14285714 83.33333333 84.52380952 86.9047619  86.9047619
#  86.9047619  86.9047619  86.9047619  88.0952381  89.28571429 89.28571429
#  89.28571429 91.66666667 91.66666667 91.66666667 91.66666667 91.66666667
#  91.66666667]


#################################################################MInkLoc++  Start ######################################################################################################
####Provided pre-trained model PCS+RGB
#python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/minklocmultimodal.txt --weights ./weights/minklocmultimodal_baseline.pth
#quantize
# Avg. top 1% recall: 77.38   Avg. similarity: 67.8590   Avg. recall @N:
# [76.19047619 77.38095238 77.38095238 79.76190476 86.9047619  90.47619048
#  90.47619048 91.66666667 91.66666667 92.85714286 92.85714286 92.85714286
#  92.85714286 92.85714286 92.85714286 92.85714286 92.85714286 92.85714286
#  92.85714286 92.85714286 92.85714286 92.85714286 92.85714286 92.85714286
#  92.85714286]
 
#mink_quantization_size = [2.5, 2.0, 0.42]
# Avg. top 1% recall: 73.81   Avg. similarity: 84.9746   Avg. recall @N:
# [69.04761905 73.80952381 75.         78.57142857 80.95238095 83.33333333
#  85.71428571 86.9047619  86.9047619  88.0952381  88.0952381  89.28571429
#  89.28571429 89.28571429 91.66666667 92.85714286 92.85714286 92.85714286
#  92.85714286 94.04761905 94.04761905 95.23809524 95.23809524 95.23809524
#  95.23809524]

####implemented model PCS+RGB
#python ./eval/evaluate_kitti.py --config ./config/config_baseline_multimodal.txt --model_config ./models/minklocmultimodal.txt --weights ./weights/model_MinkLocMultimodal_20231128_1008_final.pth
# Dataset: KITTI
# Avg. top 1% recall: 77.38   Avg. similarity: 68.5943   Avg. recall @N:
# [75.         77.38095238 82.14285714 86.9047619  89.28571429 90.47619048
#  90.47619048 90.47619048 91.66666667 91.66666667 91.66666667 92.85714286
#  92.85714286 92.85714286 92.85714286 92.85714286 92.85714286 92.85714286
#  92.85714286 92.85714286 92.85714286 92.85714286 92.85714286 92.85714286
#  92.85714286]
 
#mink_quantization_size = [2.5, 2.0, 0.42]
# Avg. top 1% recall: 75.00   Avg. similarity: 63.8633   Avg. recall @N:
# [71.42857143 75.         79.76190476 80.95238095 82.14285714 82.14285714
#  85.71428571 86.9047619  88.0952381  88.0952381  88.0952381  88.0952381
#  89.28571429 90.47619048 91.66666667 91.66666667 91.66666667 92.85714286
#  92.85714286 94.04761905 94.04761905 94.04761905 94.04761905 94.04761905
#  94.04761905]

###implemented model：PCS
#python ./eval/evaluate_kitti.py --config ./config/config_baseline.txt --model_config ./models/minkloc3d.txt --weights ./weights/model_MinkLoc3D_20231128_1621_final.pth
# Avg. top 1% recall: 60.71   Avg. similarity: 63.8046   Avg. recall @N:
# [55.95238095 60.71428571 63.0952381  65.47619048 65.47619048 70.23809524
#  70.23809524 70.23809524 75.         76.19047619 77.38095238 79.76190476
#  82.14285714 83.33333333 84.52380952 85.71428571 88.0952381  89.28571429
#  90.47619048 90.47619048 91.66666667 91.66666667 91.66666667 91.66666667
#  91.66666667]

#mink_quantization_size = [2.5, 2.0, 0.42]
# Avg. top 1% recall: 72.62   Avg. similarity: 64.7754   Avg. recall @N:
# [64.28571429 72.61904762 84.52380952 86.9047619  88.0952381  88.0952381
#  90.47619048 90.47619048 90.47619048 92.85714286 92.85714286 92.85714286
#  92.85714286 92.85714286 94.04761905 94.04761905 94.04761905 94.04761905
#  94.04761905 94.04761905 94.04761905 94.04761905 94.04761905 95.23809524
#  95.23809524]

# #base:
# #Avg. top 1% recall: 64.29   Avg. similarity: 61.7801   Avg. recall @N:
# [60.71428571 64.28571429 67.85714286 70.23809524 75.         76.19047619
#  79.76190476 82.14285714 84.52380952 84.52380952 85.71428571 86.9047619
#  89.28571429 91.66666667 92.85714286 92.85714286 92.85714286 94.04761905
#  95.23809524 95.23809524 96.42857143 96.42857143 96.42857143 96.42857143
#  96.42857143]

###implemented model：RGB
# python ./eval/evaluate_kitti.py --config ./config/config_baseline_rgb.txt --model_config ./models/minklocrgb.txt --weights ./weights/minkloc3d_baseline.pth
#model_MinkLocRGB_20231129_0329_final.pth
#mink_quantization_size = [2.5, 2.0, 0.42]
# # Avg. top 1% recall: 73.81   Avg. similarity: 19.9321   Avg. recall @N:
# # [71.42857143 73.80952381 76.19047619 78.57142857 78.57142857 80.95238095
# #  82.14285714 83.33333333 83.33333333 83.33333333 83.33333333 83.33333333
# #  83.33333333 83.33333333 83.33333333 83.33333333 84.52380952 85.71428571
# #  86.9047619  88.0952381  89.28571429 90.47619048 90.47619048 91.66666667
# #  92.85714286]


# python ./eval/evaluate_kitti.py --config ./config/config_baseline_rgb.txt --model_config ./models/minklocrgb.txt --weights ./weights/model_MinkLocRGB_20240104_1913_final.pth > ./logs/kitti/model_MinkLocRGB_20240104_1913.log
# Avg. top 1% recall: 73.81   Avg. similarity: 23.1697   Avg. recall @N:
# [71.42857143 73.80952381 73.80952381 76.19047619 76.19047619 79.76190476
#  79.76190476 79.76190476 79.76190476 79.76190476 79.76190476 80.95238095
#  83.33333333 83.33333333 84.52380952 84.52380952 84.52380952 84.52380952
#  84.52380952 84.52380952 84.52380952 84.52380952 85.71428571 88.0952381
#  88.0952381 ]


##no svt branches:
#/opt/conda/envs/name/bin/python ./eval/evaluate_kitti.py --config ./config/config_baseline_rgb.txt --model_config ./models/minklocrgb.txt --weights ./weights/model_WaveRGB_20231201_2142_final.pth >./logs/kitti/model_WaveRGB_20231201_2142.log

#################################################################MInkLoc++  End ######################################################################################################