# Author: Jacek Komorowski
# Warsaw University of Technology

import numpy as np
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME

from datasets.oxford import OxfordDataset
from datasets.augmentation import TrainTransform, TrainSetTransform, TrainRGBTransform, ValRGBTransform
from datasets.samplers import BatchSampler
from misc.utils import MinkLocParams


def make_datasets(params: MinkLocParams, debug=False):
    # Create training and validation datasets
    datasets = {}
    train_transform = TrainTransform(params.aug_mode)
    train_set_transform = TrainSetTransform(params.aug_mode)

    if params.use_rgb:
        image_train_transform = TrainRGBTransform(params.aug_mode)
        image_val_transform = ValRGBTransform()
    else:
        image_train_transform = None
        image_val_transform = None

    datasets['train'] = OxfordDataset(params.dataset_folder, params.train_file, image_path=params.image_path,
                                      lidar2image_ndx_path=params.lidar2image_ndx_path, transform=train_transform,
                                      set_transform=train_set_transform, image_transform=image_train_transform,
                                      use_cloud=params.use_cloud)
    val_transform = None
    if params.val_file is not None:
        datasets['val'] = OxfordDataset(params.dataset_folder, params.val_file, image_path=params.image_path,
                                        lidar2image_ndx_path=params.lidar2image_ndx_path, transform=val_transform,
                                        set_transform=train_set_transform, image_transform=image_val_transform,
                                        use_cloud=params.use_cloud)
    return datasets


def make_collate_fn(dataset: OxfordDataset, mink_quantization_size=None, batch_split_size=None):
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        # Constructs a batch object
        labels = [e['ndx'] for e in data_list]

        # Compute positives and negatives mask
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and
        # negatives_mask which are batch_size x batch_size boolean tensors
        result = {'positives_mask': positives_mask, 'negatives_mask': negatives_mask}

        if 'cloud' in data_list[0] and 'image' in data_list[0]:
            clouds = [e['cloud'] for e in data_list]
            clouds = torch.stack(clouds, dim=0)       # Produces (batch_size, n_points, 3) tensor
            if dataset.set_transform is not None:
                # Apply the same transformation on all dataset elements
                clouds = dataset.set_transform(clouds)

            images = [e['image'] for e in data_list]
            coords = [ME.utils.sparse_quantize(coordinates=e, quantization_size=mink_quantization_size)
                      for e in clouds]
            
            if batch_split_size is None or batch_split_size == 0:
                coords = ME.utils.batched_coordinates(coords)
                # Assign a dummy feature equal to 1 to each point
                feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
                imgs= torch.stack(images, dim=0)  
                batch = {'coords': coords, 'features': feats, 'images':imgs}
            else:
                # Split the batch into chunks
                batch = []
                for i in range(0, len(coords), batch_split_size):
                    temp = coords[i:i + batch_split_size]
                    c = ME.utils.batched_coordinates(temp)
                    f = torch.ones((c.shape[0], 1), dtype=torch.float32)
                    imgs = torch.stack(images[i:i + batch_split_size], dim=0) 
                    minibatch = {'coords': c, 'features': f, 'images':imgs}
                    batch.append(minibatch)

        else:
            if 'cloud' in data_list[0]:
                clouds = [e['cloud'] for e in data_list]
                clouds = torch.stack(clouds, dim=0)       # Produces (batch_size, n_points, 3) tensor
                if dataset.set_transform is not None:
                    # Apply the same transformation on all dataset elements
                    clouds = dataset.set_transform(clouds)
                coords = [ME.utils.sparse_quantize(coordinates=e, quantization_size=mink_quantization_size)
                          for e in clouds]

                if batch_split_size is None or batch_split_size == 0:
                    coords = ME.utils.batched_coordinates(coords)
                    # Assign a dummy feature equal to 1 to each point
                    feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
                    batch = {'coords': coords, 'features': feats}
                else:
                    # Split the batch into chunks
                    batch = []
                    for i in range(0, len(coords), batch_split_size):
                        temp = coords[i:i + batch_split_size]
                        c = ME.utils.batched_coordinates(temp)
                        f = torch.ones((c.shape[0], 1), dtype=torch.float32)
                        minibatch = {'coords': c, 'features': f}
                        batch.append(minibatch)
                    
            if 'image' in data_list[0]:
                images = [e['image'] for e in data_list]
                if batch_split_size is None or batch_split_size == 0:
                    batch['images'] = torch.stack(images, dim=0)       # Produces (N, C, H, W) tensor
                else:
                    # Split the batch into chunks
                    batch = []
                    for i in range(0, len(images), batch_split_size):
                        temp = images[i:i + batch_split_size]
                        minibatch['images'] = torch.stack(temp, dim=0)       # Produces (N, C, H, W) tensor
                        batch.append(minibatch)


        return batch, positives_mask, negatives_mask
    
    return collate_fn


def make_dataloaders(params: MinkLocParams, debug=False):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(params, debug=debug)

    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], batch_size=params.batch_size,
                                 batch_size_limit=params.batch_size_limit,
                                 batch_expansion_rate=params.batch_expansion_rate)
    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_collate_fn(datasets['train'],  params.model_params.mink_quantization_size, params.batch_split_size)
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler, collate_fn=train_collate_fn,
                                     num_workers=params.num_workers, pin_memory=True)

    if 'val' in datasets:
        val_sampler = BatchSampler(datasets['val'], batch_size=params.val_batch_size)
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        val_collate_fn = make_collate_fn(datasets['val'], params.model_params.mink_quantization_size)
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=params.num_workers, pin_memory=True)

    return dataloders


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e
