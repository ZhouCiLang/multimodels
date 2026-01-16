"""
Computes mapping between RobotCar Seasons image path and a corresponding point cloud path
Modified by ZRN
"""

import os
import pickle
import tqdm

from robotcar_seasons_benchmark.robotcar_seasons import RobotCarSeasonsDataset
from robotcar_seasons_benchmark.robotcar import RobotCarDataset


if __name__ == '__main__':
    seasons_dataset_root = '/userhome/datasets/robotcar-seasons/'
    seasons_ds = RobotCarSeasonsDataset(seasons_dataset_root)
    robotcar_root = '/userhome/datasets/RobotCar_only_cloud/'
#     robotcar_ds = RobotCarDataset(robotcar_root)
    pointcloud_l = []
    count_missing_clouds = 0
    clouds=0
    # Split image timestamps into traversals

    for image_ts in tqdm.tqdm(seasons_ds.index_ts_rel_image):
        if image_ts[0:6] =='141690': #rain
            traversal='2014-11-25-09-18-32'
        elif image_ts[0:6] =='144741':#winter
            traversal='2015-11-13-10-28-08'
        elif image_ts[0:6] =='142295':#snow
            traversal='2015-02-03-08-45-10'
        elif image_ts[0:6] =='142599':#sun
            traversal='2015-03-10-14-18-10'
        elif image_ts[0:6] =='143229':#summer
            traversal='2015-05-22-11-14-30'
        elif image_ts[0:6] =='142445':#dusk
            traversal='2015-02-20-16-34-06'
        elif image_ts[0:6] =='141884':#night-rain
            traversal='2014-12-17-18-18-43'
        elif image_ts[0:6] =='141872':#dawn
            traversal='2014-12-16-09-14-09'
        elif image_ts[0:6] =='141717':#reference
            traversal='2014-11-28-12-07-13'
        elif image_ts[0:6] =='141823':#night
            traversal='2014-12-10-18-10-50'
#         traversal = robotcar_ds.get_traversal(image_ts)
        rel_image_path = seasons_ds.index_ts_rel_image[image_ts]
        rel_cloud_path = os.path.join(traversal, 'pointclouds', str(image_ts) + '.bin')
#         abs_cloud_path = os.path.join(robotcar_root, rel_cloud_path)
        abs_cloud_path = os.path.join(robotcar_root, rel_cloud_path)        
#         print('cloud path:\n', abs_cloud_path)
        if not os.path.exists(abs_cloud_path):
            print('image path: \n', rel_image_path)
            print('Missing point cloud: {}'.format(abs_cloud_path))
            count_missing_clouds += 1
            continue

        pointcloud_l.append((image_ts, rel_image_path, rel_cloud_path))
        clouds+=1

    print('{} missing point clouds'.format(count_missing_clouds))
    print('\n point clouds:', clouds)
    pickle.dump(pointcloud_l, open("season_scans.pickle", "wb"))
