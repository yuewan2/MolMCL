import os
import math
import copy
import pickle
import numpy as np
import pandas as pd
import argparse
import rdkit
import yaml
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset
from torch_geometric.utils import to_dense_batch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from molmcl.pretrain.loader import MoleculeDataset
from molmcl.pretrain.model import GNNWrapper
from molmcl.models.gnn import GNN, GPS
from molmcl.utils.data import list_collate_fn


def train_one_epoch(rank, model, loader, optimizer, config, verbose=False, enable_kg=True):
    model.train()

    accum_iter = config['optim']['accum_iter']
    if verbose:
        pbar = tqdm(enumerate(loader), total=len(loader))
    else:
        pbar = enumerate(loader)

    loss_1_history, loss_2_history, loss_3_history, loss_4_history, loss_total_history = [], [], [], [], []
    for batch_idx, batch in pbar:
        batch = batch.to(rank)
        with torch.set_grad_enabled(True):
            output = model(batch)
            loss = output['loss_1'] + output['loss_2'] + output['loss_3'] + \
                config['optim']['reg_coeff'] * (output['loss_1_reg'] + output['loss_2_reg'])

            if config['optim']['knowledge_guided']:
                loss += int(enable_kg) * config['optim']['reg_coeff'] * output['loss_4']
                loss_4 = output['loss_4'].item()
            else:
                loss_4 = 0

            loss_1_history.append(output['loss_1'].item())
            loss_2_history.append(output['loss_2'].item())
            loss_3_history.append(output['loss_3'].item())
            loss_4_history.append(loss_4)
            loss_total_history.append(loss.item())

            loss = loss / accum_iter
            loss.backward()

            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(loader)):
                optimizer.step()
                optimizer.zero_grad()

    return loss_1_history, loss_2_history, loss_3_history, loss_4_history, loss_total_history


def main(config):
    rank = int(os.environ['LOCAL_RANK'])
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl')

    dataset = MoleculeDataset(config['dataset']['data_dir'],
                              num_candidates=config['optim']['num_candidates'],
                              feat_type=config['dataset']['feat_type'])
    
    if rank == 0:
        print(config)
        print('Dataset size:', len(dataset))
        print('Example:')
        for data in dataset[0]:
            print('smi={}'.format(data.smi))

    # Setup dataloader
    loader = DataLoader(dataset, batch_size=config['batch_size'],
                        num_workers=config['dataset']['num_workers'],
                        shuffle=False, collate_fn=list_collate_fn,
                        sampler=DistributedSampler(dataset))

    # Setup model
    if config['dataset']['feat_type'] == 'basic':
        atom_feat_dim, bond_feat_dim = None, None
    elif config['dataset']['feat_type'] == 'rich':
        atom_feat_dim, bond_feat_dim = 143, 14
    elif config['dataset']['feat_type'] == 'super_rich':
        atom_feat_dim, bond_feat_dim = 170, 14
    else:
        raise NotImplementedError('Unrecognized feature type. Please choose from [basic/rich/super_rich].')

    if config['model']['backbone'] == 'gin':
        gnn = GNN(num_layer=config['model']['num_layer'],
                  emb_dim=config['model']['emb_dim'],
                  drop_ratio=config['model']['dropout_ratio'],
                  atom_feat_dim=atom_feat_dim,
                  bond_feat_dim=bond_feat_dim)
    elif config['model']['backbone'] == 'gps':
        gnn = GPS(channels=config['model']['emb_dim'], pe_dim=20,
                  node_dim=atom_feat_dim,
                  edge_dim=bond_feat_dim,
                  num_layers=config['model']['num_layer'],
                  heads=config['model']['heads'], 
                  attn_type='multihead',
                  dropout=config['model']['dropout_ratio'],
                  attn_dropout=config['model']['dropout_ratio'])
    else:
        raise NotImplementedError

    model = GNNWrapper(gnn, config['model']['emb_dim'], config,
                       atom_context_size=len(dataset.atom_vocab_itos),
                       layer_norm_out=True)

    model = DDP(model.to(rank), device_ids=[rank], find_unused_parameters=False)

    if config['model']['checkpoint']:
        print('Loading wrapper checkpoint from {}'.format(config['model']['checkpoint']))
        loaded_state_dict = torch.load(config['model']['checkpoint'])['wrapper']
        model.module.load_state_dict(loaded_state_dict, strict=True)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['optim']['lr'], weight_decay=float(config['optim']['decay']))

    # Pretrain
    global_train_loss = float('inf')
    global_eval_loss = float('inf')
    if rank == 0:
        pbar = tqdm(range(config['start_epoch'] + 1, config['epochs'] + 1))
    else:
        pbar = range(config['start_epoch'] + 1, config['epochs'] + 1)

    for epoch in pbar:
        loader.sampler.set_epoch(epoch)
        loss_1, loss_2, loss_3, loss_4, loss_total = \
            train_one_epoch(rank, model, loader, optimizer, config, verbose=rank == 0, enable_kg=epoch>20)

        if rank == 0:
            pbar.set_description('[Epoch {}] loss={} #1={} #2={} #3={} #4={}'.format(
                epoch, round(np.mean(loss_total), 5), round(np.mean(loss_1), 5),
                round(np.mean(loss_2), 5), round(np.mean(loss_3), 5), round(np.mean(loss_4), 5)))

            if np.mean(loss_total) < global_train_loss:
                global_train_loss = np.mean(loss_total)
                checkpoint_path = '{}/zinc-{}-best.pt'.format(config['output_dir'], config['method'])
                torch.save({'wrapper': model.module.state_dict()}, checkpoint_path)
                print('Save to {} at epoch {}'.format(checkpoint_path, epoch))

            checkpoint_path = '{}/zinc-{}-last.pt'.format(config['output_dir'], config['method'])
            torch.save({'wrapper': model.module.state_dict()}, checkpoint_path)

            if epoch % 2 == 0:
                checkpoint_path = '{}/zinc-{}-ep{}.pt'.format(config['output_dir'], config['method'], epoch)
                torch.save({'wrapper': model.module.state_dict()}, checkpoint_path)

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    with open('./config/pretrain.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(config)