# Author: Jacek Komorowski, Monika Wysoczanska
# Warsaw University of Technology
# Modified by: Kamil Zywanowski, Adam Banaszczyk, Michal Nowicki (Poznan University of Technology 2021)

# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import argparse
import torch
import MinkowskiEngine as ME
import tqdm

from misc.utils import MinkLocParams
from models.model_factory import model_factory
from PIL import Image
from datasets.augmentation import ValRGBTransform

# from datasets.dataset_utils import to_spherical


DEBUG = False


def evaluate(model, device, params, silent=True):
    # Run evaluation on all eval datasets

    if DEBUG:
        params.eval_database_files = params.eval_database_files[0:1]
        params.eval_query_files = params.eval_query_files[0:1]

    assert len(params.eval_database_files) == len(params.eval_query_files)

    stats = {}
    for database_file, query_file in zip(params.eval_database_files, params.eval_query_files):
        # Extract location name from query and database files
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(params.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)
#             database_sets = pickle.load(open('KITTI_00_query_samp10.pickle','rb'))
        p = os.path.join(params.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        temp = evaluate_dataset(model, device, params, database_sets, query_sets, silent=silent)
        stats[location_name] = temp

    return stats


def evaluate_dataset(model, device, params, database_sets, query_sets, silent=True):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    model.eval()

    for set in tqdm.tqdm(database_sets, disable=silent):
        database_embeddings.append(get_latent_vectors(model, set, device, params))

    for set in tqdm.tqdm(query_sets, disable=silent):
        query_embeddings.append(get_latent_vectors(model, set, device, params))

    for i in tqdm.tqdm(range(len(query_sets)), disable=silent):
        for j in range(len(query_sets)):
            pair_recall, pair_similarity, pair_opr = get_recall(i, j, database_embeddings, query_embeddings, query_sets)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    ave_recall = recall / count
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall,
             'average_similarity': average_similarity}
    return stats


def load_data_item(filename, params):
    # Load point cloud, does not apply any transform
    # Returns Nx3 matrix or Nx4 matrix depending on the intensity value
    file_path = os.path.join(params.dataset_folder, filename["query_velo"])
    result = {}
    if params.use_cloud:
        pc = np.fromfile(file_path, dtype=np.float32).reshape([-1, 4])
#         print('filename:\n', filename["query_velo"])
#         print('pc.shape\n',pc.shape)
        pc = pc[:, :3]
#         print('pc shape:\n', pc.shape)
#         # coords are within -1..1 range in each dimension
#         assert pc.shape[0] == params.num_points * 3, "Error in point cloud shape: {}".format(file_path)
#         pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        pc = torch.tensor(pc, dtype=torch.float)
        result['coords'] = pc

    if params.use_rgb: ####need to change here:
        image_file_path =os.path.join(params.dataset_folder, filename["query_img"])
        img = Image.open(image_file_path)
        transform = ValRGBTransform()
        # Convert to tensor and normalize
        result['image'] = transform(img)
# 299: {'query_velo': 'sequences/00/velodyne/004520.bin', 'query_img': 'sequences/00/image_2/004520.png', 0: [6, 7, 8, 9, 10, 11, 12, 13, 14, 151, 152, 153, 154, 150]}, 300: {'query_velo': 'sequences/00/velodyne/004529.bin', 'query_img': 'sequences/00/image_2/004529.png', 0: [8, 9, 10, 11, 12, 13, 14, 15, 155, 151, 152, 153, 154, 149, 150]}, 301: {'query_velo': 'sequences/00/velodyne/004537.bin', 'query_img': 'sequences/00/image_2/004537.png', 0: [8, 9, 10, 11, 12, 13, 14, 15, 155, 151, 152, 153, 154, 148, 149, 150]}
    return result

def get_latent_vectors(model, set, device, params):
    # Adapted from original PointNetVLAD code

    """
    if DEBUG:
        embeddings = torch.randn(len(set), 256)
        return embeddings
    """

    if DEBUG:
        embeddings = np.random.rand(len(set), 256)
        return embeddings

    model.eval()
    embeddings_l = []
    for elem_ndx in tqdm.tqdm(set):
#         print('set[elem_ndx]\n', set[elem_ndx])
        x = load_data_item(set[elem_ndx], params)
        with torch.no_grad():
            
            # coords are (n_clouds, num_points, channels) tensor
            batch = {}
            if params.use_cloud:
                coords = ME.utils.sparse_quantize(coordinates=x['coords'],
                                                  quantization_size=params.model_params.mink_quantization_size)
                bcoords = ME.utils.batched_coordinates([coords]).to(device)
                # Assign a dummy feature equal to 1 to each point
                feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32).to(device)
                batch['coords'] = bcoords
                batch['features'] = feats

            if params.use_rgb:
                batch['images'] = x['image'].unsqueeze(0).to(device)

            x = model(batch)
            embedding = x['embedding']
            
            if params.normalize_embeddings:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)  # Normalize embeddings

        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)

    embeddings = np.vstack(embeddings_l)
    return embeddings


def get_recall(m, n, database_vectors, query_vectors, query_sets):
    # based on original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
    recall = (np.cumsum(recall) / float(num_evaluated)) * 100
    return recall, top1_similarity_score, one_percent_recall


def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall'], stats[database_name]['average_similarity']))
        print(stats[database_name]['ave_recall'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on KITTI dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')

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

    params.eval_database_files = ["KITTI_00_database_samp10.pickle"]
    params.eval_query_files = ["KITTI_00_query_samp10.pickle"]
    params.dataset_folder = "/userhome/datasets/Kitti_Odometry/dataset/"
#     params.image_path="/datasets/Kitti_Odometry/dataset/sequences/00/"
    params.dataset_name = "KITTI"
    params.model_params.mink_quantization_size = [2.5, 2.0, 0.42]
    params.print()
    print('#'*30)
    print('WARNING: Database and query files, paths and quantization from config are overwritten by KITTI specs in evaluate_kitti.py.')
    print('#'*30, '\n')

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

    stats = evaluate(model, device, params, silent=False)
    print_eval_stats(stats)
