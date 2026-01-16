"""
Estimate approximate 6DOF poses of elements in the RobotCar Seasons query set, by searching for the closest embedding
in the database (reference traversal) and returning its pose.
Poses are saved to the output file in the format required by RobotCar Seasons submission website:
https://www.visuallocalization.net/submission/
"""

import os
import pickle
import numpy as np
import random
from scipy.spatial.transform import Rotation
import argparse
from robotcar_seasons_benchmark.robotcar_seasons import RobotCarSeasonsDataset
import robotcar_seasons_benchmark.mvg as mvg


if __name__ == '__main__':
    # Path to the pickle with computed global descriptors for elements in the RobotCar Seasons (both query and database sets)
    data_path ='./robotcar_seasons_benchmark/season_scans.pickle'
    pickle_path = '/userhome/code/MinkLocMultimodal-master/robotcar_seasons_benchmark/season_scan_embeddings_multi_org.pickle'
    results_file = '/userhome/code/MinkLocMultimodal-master/robotcar_seasons_benchmark/season_scan_embeddings_minklocplusplus.txt'

    assert os.path.exists(pickle_path)
    data = pickle.load(open(data_path, "rb"))
    feats = pickle.load(open(pickle_path, "rb"))
    
    embeddings = {}
    for i in range(0, len(data)):
        embeddings[data[i][0]] = feats[i]
    
    seasons_dataset_root = '/userhome/datasets/robotcar-seasons'
    seasons_ds = RobotCarSeasonsDataset(seasons_dataset_root)

    # Gather embeddings of dataset (reference traversal) elements
    databaset_elements = seasons_ds.index_traversal[seasons_ds.reference_traversal_name]
    print('data:{} \t embedding:{} \t datasets:{}\t'.format(len(data), len(embeddings), len(databaset_elements)))
    
    db_ts = []      # Database element timestaps
    db_emb = []     # Database element embeddings
    missing_scans = 0
#     print('dataset:\n', databaset_elements)
#     print('embeddings:\n', embeddings)
    for query_ts in databaset_elements:
        if query_ts in embeddings:
#             print('query_ts:\n', query_ts)
            db_ts.append(query_ts)
            db_emb.append(embeddings[query_ts])
        else:
            missing_scans += 1

    db_emb = np.array(db_emb)
    print('{} missing scans in reference traversal'.format(missing_scans))
    print('db_ts len:{}\n'.format(len(db_ts)))

    threshold = 10      # 5 meter threshold for coarse localization

    with open(results_file, "w") as file:
        for traversal in seasons_ds.traversals:
            print('Traversal: {}'.format(traversal))
            missing_scans = 0
            tp = []
            if traversal == seasons_ds.reference_traversal_name:
                continue

            for query_ts in seasons_ds.index_traversal[traversal]:
                # Find the nearest neighbour
                if query_ts not in embeddings:
                    missing_scans += 1
                    # Missing point cloud, as a workaround select a random database element
                    ndx = random.randrange(len(db_emb))
#                     print('random ndx:{}\n'.format(ndx))
                else:
#                     print('db_emb:{}\n'.format(db_emb[0:2]))
#                     print('emb:{}\n'.format(embeddings[query_ts]))
#                     print('sub:{}\n'.format(db_emb[0:2]-embeddings[query_ts]))
                    x = (db_emb - embeddings[query_ts]) ** 2
                    ##Modified by ZRN: use squeeze to output the right result.
                    dist = np.sum(np.squeeze(x), axis=1)
#                     print('x:and x.len \n', x, len(x))
#                     print('dist len:\n', len(dist))
                    ndx = np.argmin(dist)
#                     print('dist{} and ndx{}\n'.format(dist, ndx))
                
                # ndx is the nearest neighbour index
                nn_ts = db_ts[ndx]      # Get nearest neighbour timestamp
                nn_pose = seasons_ds.poses[nn_ts]
                # Convert to the format required by submission website: quaternion and translation
                r = Rotation.from_matrix(nn_pose.R)
                quat = r.as_quat()
                # This returns quaternion in scalar-last mode, where qw is quat[3]
                # RobotCarSeasons expects submissions in scalar-first format
                quat = np.array([quat[3], quat[0], quat[1], quat[2]])
                t = - nn_pose.R @ nn_pose.t
                ####################################Added by ZRN for pair informations################################################
                line1 ='rear/' + nn_ts+'.jpg\n' 
                file.write(line1)
                line = traversal +'/' + query_ts + '.jpg {} {} {} {} {} {} {}\n'.format(quat[0], quat[1], quat[2], quat[3],
                                                                                 t[0, 0], t[1, 0], t[2, 0])
                 ####################################Added by ZRN for pair informations################################################                                                         
                file.write(line)

                if query_ts in seasons_ds.poses:
                    # Pose of the query image given
                    query_pose = seasons_ds.poses[query_ts]
#                     print('query pose:{}\n, nn_pose:{}\n'.format(query_pose.R, nn_pose.R))
                    rel_pose = mvg.relative_camera_pose(nn_pose, query_pose)
                    delta_t = np.linalg.norm(rel_pose.t)
                    tp.append(1 if delta_t <= threshold else 0)

#             print('{} missing scans'.format(missing_scans))
            print('Percentage of queries correctly localized ({}m threshold): {:0.3f}'.format(threshold, np.mean(tp)))

