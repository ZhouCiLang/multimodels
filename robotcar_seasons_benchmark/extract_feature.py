"""
Estimate approximate 6DOF poses of elements in the RobotCar Seasons query set, by searching for the closest embedding
in the database (reference traversal) and returning its pose.
Poses are saved to the output file in the format required by RobotCar Seasons submission website:
https://www.visuallocalization.net/submission/
"""
import argparse
import torch
import MinkowskiEngine as ME
import tqdm
import time

import os
import pickle
import numpy as np
import random

from robotcar_seasons_benchmark.robotcar_seasons import RobotCarSeasonsDataset

from PIL import Image
from misc.utils import MinkLocParams
from models.model_factory import model_factory
from datasets.augmentation import ValRGBTransform

def load_data_item(image_path, point_path, params):
    # returns Nx3 matrix

    result = {}

    pc = np.fromfile(point_path, dtype=np.float64)
    # coords are within -1..1 range in each dimension
    pc = np.reshape(pc, (pc.shape[0] // 3, 3))
    pc = torch.tensor(pc, dtype=torch.float)
    result['coords'] = pc

    img = Image.open(image_path)
    transform = ValRGBTransform()
    # Convert to tensor and normalize
    result['image'] = transform(img)

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model on RobotCar dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
#     parser.add_argument('--feature_path', type=str, default='./robotcar_seasons_benchmark/season_scan_embeddings_multi_pc_org1.pickle', help='Path of feature path')
    parser.add_argument('--weights', type=str, required=True, help='Trained model weights')

    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    print('')

    params = MinkLocParams(args.config, args.model_config)
    params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(params)
    
    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)
    model.eval()
    # Path to the pickle with computed global descriptors for elements in the RobotCar Seasons (both query and database sets)
    data_path = './robotcar_seasons_benchmark/season_scans.pickle'
    #weights=.'/weights/model_WaveMultimodalInferAll_20231222_2043_final.pth'
    #./robotcar_seasons_benchmark/season_scan_embeddings_multi_pc_org1.pickle
    feature_path = './robotcar_seasons_benchmark/'+ args.weights[10:-3]+'pickle'
    data = pickle.load(open(data_path, "rb"))
    image_dataset_root = '/userhome/datasets/robotcar-seasons/'
    pc_dataset_root = '/userhome/datasets/RobotCar_only_cloud/'
    
    feature = []
    
    #Load models here:
    
    for i in range(0, len(data)):
        image=os.path.join(image_dataset_root, data[i][1])
        point=os.path.join(pc_dataset_root, data[i][2])
        
        x = load_data_item(image, point, params)
        batch = {}
        
        coords = ME.utils.sparse_quantize(coordinates=x['coords'], quantization_size=params.model_params.mink_quantization_size)
        bcoords = ME.utils.batched_coordinates([coords]).to(device)
        # Assign a dummy feature equal to 1 to each point
        feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32).to(device)
        batch['coords'] = bcoords
        batch['features'] = feats
        batch['images'] = x['image'].unsqueeze(0).to(device)
         
        x= model(batch)
        embedding = x['embedding']
        embedding = embedding.detach().cpu().numpy()
        feature.append(embedding)
        
    pickle.dump(feature, open(feature_path, "wb"))
