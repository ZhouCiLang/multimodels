# Author: Jacek Komorowski
# Warsaw University of Technology

# Train on Oxford dataset (from PointNetVLAD paper) using BatchHard hard negative mining.

import os
from datetime import datetime
import numpy as np
import torch
import pickle
import tqdm
import pathlib
import wandb

# from torch.utils.tensorboard import SummaryWriter

from eval.evaluate import evaluate, print_eval_stats
from misc.utils import MinkLocParams, get_datetime
from models.loss import make_losses
from models.model_factory import model_factory
from models.minkloc_multimodal import MinkLocMultimodal


VERBOSE = False


def print_stats(stats, phase):
    if 'num_triplets' in stats:
        # For triplet loss
        s = '{} - Loss (mean/total): {:.4f} / {:.4f}    Avg. embedding norm: {:.4f}   Triplets per batch (all/non-zero): {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['total_loss'], stats['avg_embedding_norm'], stats['num_triplets'],
                       stats['num_non_zero_triplets']))
    elif 'num_pos' in stats:
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   #positives/negatives: {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pos'], stats['num_neg']))

    s = ''
    l = []
    if 'mean_pos_pair_dist' in stats:
        s += 'Pos dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}   Neg dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}'
        l += [stats['min_pos_pair_dist'], stats['mean_pos_pair_dist'], stats['max_pos_pair_dist'],
              stats['min_neg_pair_dist'], stats['mean_neg_pair_dist'], stats['max_neg_pair_dist']]
    if 'pos_loss' in stats:
        if len(s) > 0:
            s += '   '
        s += 'Pos loss: {:.4f}  Neg loss: {:.4f}'
        l += [stats['pos_loss'], stats['neg_loss']]
    if 'recall' in stats:
        s += f"Recall@1: {stats['recall'][1]:.4f}   "
    if 'ap' in stats:
        s += f"AP: {stats['ap']:.4f}   "

    if len(l) > 0:
        print(s.format(*l))

    if 'final_loss' in stats:
        # Multi loss
        s1 = '{} - Loss (total/final'.format(phase)
        s2 = '{:.4f} / {:.4f}'.format(stats['loss'], stats['final_loss'])
        s3 = 'Active triplets (final '
        if 'final_num_non_zero_triplets' in stats:
            s4 = '{:.1f}'.format(stats['final_num_non_zero_triplets'])
        if 'cloud_loss' in stats:
            s1 += '/cloud'
            s2 += '/ {:.4f}'.format(stats['cloud_loss'])
            s3 += '/cloud'
            if 'final_num_non_zero_triplets' in stats:
                s4 += '/ {:.1f}'.format(stats['cloud_num_non_zero_triplets'],)
        if 'image_loss' in stats:
            s1 += '/image'
            s2 += '/ {:.4f}'.format(stats['image_loss'])
            s3 += '/image'
            if 'final_num_non_zero_triplets' in stats:
                s4 += '/ {:.1f}'.format(stats['image_num_non_zero_triplets'],)

        s1 += '): '
        s3 += '): '
        print(s1 + s2)
        if 'final_num_non_zero_triplets' in stats:  ##Added by ZRN
            print(s3 + s4)
    
def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats

def training_step(global_iter, model, phase, device, optimizer, loss_fn):
    assert phase in ['train', 'val']

    batch, positives_mask,  negatives_mask= next(global_iter)
    batch = {e: batch[e].to(device) for e in batch}
    
    if phase == 'train':
        model.train()
    else:
        model.eval()

    optimizer.zero_grad()

    with torch.set_grad_enabled(phase == 'train'):
        embeddings = model(batch)
        stats = model.stats.copy() if hasattr(model, 'stats') else {}

        loss, temp_stats = loss_fn(embeddings, positives_mask, negatives_mask)
        temp_stats = tensors_to_numbers(temp_stats)
        stats.update(temp_stats)
        stats['loss']=loss.item()
        if phase == 'train':
            loss.backward()
            optimizer.step()

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    return stats


def multistaged_training_step(global_iter, model, phase, device, optimizer, loss_fn):
    # Training step using multistaged backpropagation algorithm as per:
    # "Learning with Average Precision: Training Image Retrieval with a Listwise Loss"
    # This method will break when the model contains Dropout, as the same mini-batch will produce different embeddings.
    # Make sure mini-batches in step 1 and step 3 are the same (so that BatchNorm produces the same results)
    # See some exemplary implementation here: https://gist.github.com/ByungSun12/ad964a08eba6a7d103dab8588c9a3774

    assert phase in ['train', 'val']
    batch, positives_mask, negatives_mask= next(global_iter)
    
    if phase == 'train':
        model.train()
    else:
        model.eval()

    # Stage 1 - calculate descriptors of each batch element (with gradient turned off)
    # In training phase network is in the train mode to update BatchNorm stats
    embeddings_l = []
    with torch.set_grad_enabled(False):
        for minibatch in batch:
            minibatch = {e: minibatch[e].to(device) for e in minibatch}
            y = model(minibatch)
            embeddings_l.append(y)

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    # Stage 2 - compute gradient of the loss w.r.t embeddings
    embeddings = torch.cat(embeddings_l, dim=0)

    with torch.set_grad_enabled(phase == 'train'):
        if phase == 'train':
            embeddings.requires_grad_(True)
        loss, stats = loss_fn(embeddings, positives_mask, negatives_mask)
        stats = tensors_to_numbers(stats)
        if phase == 'train':
            loss.backward()
            embeddings_grad = embeddings.grad

    # Delete intermediary values
    embeddings_l, embeddings, y, loss = None, None, None, None

    # Stage 3 - recompute descriptors with gradient enabled and compute the gradient of the loss w.r.t.
    # network parameters using cached gradient of the loss w.r.t embeddings
    if phase == 'train':
        optimizer.zero_grad()
        i = 0
        with torch.set_grad_enabled(True):
            for minibatch in batch:
                minibatch = {e: minibatch[e].to(device) for e in minibatch}
                embeddings = model(minibatch)
                minibatch_size = len(embeddings)
                # Compute gradients of network params w.r.t. the loss using the chain rule (using the
                # gradient of the loss w.r.t. embeddings stored in embeddings_grad)
                # By default gradients are accumulated
                embeddings.backward(gradient=embeddings_grad[i: i+minibatch_size])
                i += minibatch_size

            optimizer.step()

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    return stats

def do_train(dataloaders, params: MinkLocParams, debug=False):
    # Create model class
    s = get_datetime()
    model = model_factory(params)
    model_name = 'model_' + params.model_params.model + '_' + s
    print('Model name: {}'.format(model_name))
    weights_path = create_weights_folder()
    model_pathname = os.path.join(weights_path, model_name)
    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum([param.nelement() for param in model.parameters()])
        print('Number of model parameters: {}'.format(n_params))

    # Move the model to the proper device before configuring the optimizer
    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)
    else:
        device = "cpu"

    print('Model device: {}'.format(device))

    loss_fn = make_losses(params)

    params_l = []
    if isinstance(model, MinkLocMultimodal):
        # Different LR for image feature extractor (pretrained ResNet)
        if model.image_fe is not None:
            params_l.append({'params': model.image_fe.parameters(), 'lr': params.image_lr})
        if model.cloud_fe is not None:
            params_l.append({'params': model.cloud_fe.parameters(), 'lr': params.lr})
        if model.final_block is not None:
            params_l.append({'params': model.final_net.parameters(), 'lr': params.lr})
    else:
        # All parameters use the same lr
        params_l.append({'params': model.parameters(), 'lr': params.lr})

    # Training elements
    if params.optimizer == 'Adam':
        if params.weight_decay is None or params.weight_decay == 0:
            optimizer = torch.optim.Adam(params_l)
        else:
            optimizer = torch.optim.Adam(params_l, weight_decay=params.weight_decay)
    elif params.optimizer == 'SGD':
        # SGD with momentum (default momentum = 0.9)
        if params.weight_decay is None or params.weight_decay == 0:
            optimizer = torch.optim.SGD(params_l)
        else:
            optimizer = torch.optim.SGD(params_l, weight_decay=params.weight_decay)
    else:
        raise NotImplementedError('Unsupported optimizer: {}'.format(params.optimizer))
    if params.scheduler is None:
        scheduler = None
    else:
        if params.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs+1,
                                                                   eta_min=params.min_lr)
        elif params.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.scheduler_milestones, gamma=0.1)
        else:
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(params.scheduler))
    ###Modified by ZRN1210############################################################################################
    if params.batch_split_size is None or params.batch_split_size == 0:
        train_step_fn = training_step
    else:
        # Multi-staged training approach with large batch split into multiple smaller chunks with batch_split_size elems
        train_step_fn = multistaged_training_step
    #################################################################################################################
    
        
    ###########################################################################
    # Initialize TensorBoard writer
    ###########################################################################

#     now = datetime.now()
#     logdir = os.path.join("../tf_logs", now.strftime("%Y%m%d-%H%M%S"))
#     writer = SummaryWriter(logdir)
    
    
    params_dict = {e: params.__dict__[e] for e in params.__dict__ if e != 'model_params'}
    model_params_dict = {"model_params." + e: params.model_params.__dict__[e] for e in params.model_params.__dict__}
    params_dict.update(model_params_dict)
    wandb.init(project='MinkMultimodal', config=params_dict)
    ###########################################################################
    #
    ###########################################################################
    
    # Training statistics
    stats = {'train': [], 'val': [], 'eval': []}
    
    is_validation_set = 'val' in dataloaders
    if is_validation_set:
        phases = ['train', 'val']
    else:
        phases = ['train']

    for epoch in tqdm.tqdm(range(1, params.epochs + 1)):
        metrics = {'train': {}, 'val': {}}      # Metrics for wandb reporting
        for phase in phases:
            ####Modified by ZRN1210:###########################################################
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

            running_stats = []  # running stats for the current epoch
            count_batches = 0
            if phase == 'train':
                global_iter = iter(dataloaders['train'])
            else:
                global_iter = None if dataloaders['val'] is None else iter(dataloaders['val'])

            while True:
                count_batches += 1
                batch_stats = {}
                if debug and count_batches > 2:
                    break

                try:
                    temp_stats = train_step_fn(global_iter, model, phase, device, optimizer, loss_fn)
                    batch_stats= temp_stats

                except StopIteration:
                    # Terminate the epoch when one of dataloders is exhausted
                    break
                    
#             for batch in dataloaders[phase]:
#                 # batch is (batch_size, n_points, 3) tensor
#                 # labels is list with indexes of elements forming a batch
#                 count_batches += 1
#                 batch_stats = {}

#                 if debug and count_batches > 2:
#                     break

#                 batch = {e: batch[e].to(device) for e in batch}

#                 positives_mask = batch['positives_mask']
#                 negatives_mask = batch['negatives_mask']
#                 n_positives = torch.sum(positives_mask).item()
#                 n_negatives = torch.sum(negatives_mask).item()
#                 if n_positives == 0 or n_negatives == 0:
#                     # Skip a batch without positives or negatives
#                     print('WARNING: Skipping batch without positive or negative examples')
#                     continue

#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     # Compute embeddings of all elements
#                     embeddings = model(batch)
#                     loss, temp_stats, _ = loss_fn(embeddings, positives_mask, negatives_mask)

#                     temp_stats = tensors_to_numbers(temp_stats)
#                     batch_stats.update(temp_stats)
#                     batch_stats['loss'] = loss.item()

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

                running_stats.append(batch_stats)
#                 torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
                

            # ******* PHASE END *******
            # Compute mean stats for the phase
            epoch_stats = {}
            for key in running_stats[0].keys():
#                 temp = [e[key] for e in running_stats]
#                 epoch_stats[key] = np.mean(temp)
                ##modified by ZRN
                temp = [e[key] for e in running_stats]
                if type(temp[0]) is dict:
                    epoch_stats[key] = {key: np.mean([e[key] for e in temp]) for key in temp[0]}
                elif type(temp[0]) is np.ndarray:
                    # Mean value per vector element
                    epoch_stats[key] = np.mean(np.stack(temp), axis=0)
                else:
                    epoch_stats[key] = np.mean(temp)
                        
            stats[phase].append(epoch_stats)
            print_stats(epoch_stats, phase)
            
                        # Log metrics for wandb
            #######################################################
#             print('epoch_stats:\n', epoch_stats)
            # epoch_stats:
            # {'loss': 0.23959637978638035, 'avg_embedding_norm': 9.104822900540753, 'num_non_zero_triplets': 4.235983876877977, 'num_triplets': 7.9985342616342985, 'mean_pos_pair_dist': 0.9609062820836798, 'mean_neg_pair_dist': 1.2185259145579088, 'max_pos_pair_dist': 1.2616178278916046, 'max_neg_pair_dist': 1.6200411234115148, 'min_pos_pair_dist': 0.7164509316183526, 'min_neg_pair_dist': 0.9781598209155353, 'normalized_loss': 1.1562820092517951, 'total_loss': 1.1562820092517951}
            metrics[phase]['loss1'] = epoch_stats['loss']
#             metrics[phase]['total_loss']=epoch_stats['total_loss']
            if 'num_non_zero_triplets' in epoch_stats:
                metrics[phase]['active_triplets1'] = epoch_stats['num_non_zero_triplets']

            if 'final_loss' in epoch_stats:
                metrics[phase]['final_loss'] = epoch_stats['final_loss']

            if 'num_triplets' in epoch_stats:
                metrics[phase]['num_triplets'] = epoch_stats['num_triplets']
            
            if 'num_pairs' in epoch_stats:
                metrics[phase]['num_pairs'] = epoch_stats['pos_pairs_above_threshold']
            
            if 'recall' in epoch_stats:
                metrics[phase]['recall@1'] = epoch_stats['recall'][1]
            
            if 'ap' in epoch_stats:
                metrics[phase]['AP'] = epoch_stats['ap']

#         print('metrics:\n', metrics)
        # ******* EPOCH END *******
        wandb.log(metrics)
        
        if scheduler is not None:
            scheduler.step()

#         loss_metrics = {'train': stats['train'][-1]['loss']}
#         if 'val' in phases:
#             loss_metrics['val'] = stats['val'][-1]['loss']
#         writer.add_scalars('Loss', loss_metrics, epoch)

#         if 'num_triplets' in stats['train'][-1]:
#             nz_metrics = {'train': stats['train'][-1]['num_non_zero_triplets']}
#             if 'val' in phases:
#                 nz_metrics['val'] = stats['val'][-1]['num_non_zero_triplets']
#             writer.add_scalars('Non-zero triplets', nz_metrics, epoch)

#         elif 'num_pairs' in stats['train'][-1]:
#             nz_metrics = {'train_pos': stats['train'][-1]['pos_pairs_above_threshold'],
#                           'train_neg': stats['train'][-1]['neg_pairs_above_threshold']}
#             if 'val' in phases:
#                 nz_metrics['val_pos'] = stats['val'][-1]['pos_pairs_above_threshold']
#                 nz_metrics['val_neg'] = stats['val'][-1]['neg_pairs_above_threshold']
#             writer.add_scalars('Non-zero pairs', nz_metrics, epoch)

        if params.batch_expansion_th is not None:
            # Dynamic batch expansion of the training batch
            epoch_train_stats = stats['train'][-1]
            if 'num_non_zero_triplets' in epoch_train_stats:
                # Ratio of non-zero triplets
                rnz = epoch_train_stats['num_non_zero_triplets'] / epoch_train_stats['num_triplets']
                if rnz < params.batch_expansion_th:
                    dataloaders['train'].batch_sampler.expand_batch()
            elif 'final_num_non_zero_triplets' in epoch_train_stats:
                rnz = []
                rnz.append(epoch_train_stats['final_num_non_zero_triplets'] / epoch_train_stats['final_num_triplets'])
                if 'image_num_non_zero_triplets' in epoch_train_stats:
                    rnz.append(epoch_train_stats['image_num_non_zero_triplets'] / epoch_train_stats['image_num_triplets'])
                if 'cloud_num_non_zero_triplets' in epoch_train_stats:
                    rnz.append(epoch_train_stats['cloud_num_non_zero_triplets'] / epoch_train_stats['cloud_num_triplets'])
                rnz = max(rnz)
                if rnz < params.batch_expansion_th:
                    dataloaders['train'].batch_sampler.expand_batch()
            else:
                print('WARNING: Batch size expansion is enabled, but the loss function is not supported')

    print('')

    # Save final model weights
    final_model_path = model_pathname + '_final.pth'
    torch.save(model.state_dict(), final_model_path)

    stats = {'train_stats': stats, 'params': params}

    # Evaluate the final model
    model.eval()
    print('Evaluating the final model...')
    final_eval_stats = evaluate(model, device, params, silent=False)
    print('Final model results:')
    print_eval_stats(final_eval_stats)
    stats['eval'] = {'final': final_eval_stats}
    print('')

    # Pickle training stats and parameters
    pickle_path = model_pathname + '_stats.pickle'
    pickle.dump(stats, open(pickle_path, "wb"))

    # Append key experimental metrics to experiment summary file
    model_params_name = os.path.split(params.model_params.model_params_path)[1]
    config_name = os.path.split(params.params_path)[1]
    _, model_name = os.path.split(model_pathname)
    prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)
    export_eval_stats("experiment_results.txt", prefix, final_eval_stats)


def export_eval_stats(file_name, prefix, eval_stats):
    s = prefix
    ave_1p_recall_l = []
    ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        for ds in ['oxford', 'university', 'residential', 'business']:
            if ds not in eval_stats:
                continue
            ave_1p_recall = eval_stats[ds]['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = eval_stats[ds]['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        mean_1p_recall = np.mean(ave_1p_recall_l)
        mean_recall = np.mean(ave_recall_l)
        s += ", {:0.2f}, {:0.2f}\n".format(mean_1p_recall, mean_recall)
        f.write(s)


def create_weights_folder():
    # Create a folder to save weights of trained models
    this_file_path = pathlib.Path(__file__).parent.absolute()
    temp, _ = os.path.split(this_file_path)
    weights_path = os.path.join(temp, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    assert os.path.exists(weights_path), 'Cannot create weights folder: {}'.format(weights_path)
    return weights_path
