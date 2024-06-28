from typing import Any, Dict, Optional
from collections import defaultdict
import math
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import GINEConv, GPSConv, MessagePassing
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.utils import to_dense_batch

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add", edge_feat_dim=None):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))

        if edge_feat_dim is None:
            self.basic_feat = True
            self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
            self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)
            torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        else:
            self.basic_feat = False
            self.edge_embedding = nn.Linear(edge_feat_dim, emb_dim)
            torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)

        self.aggr = aggr

    def get_edge_embedding(self, edge_attr):
        if self.basic_feat:
            edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        else:
            edge_embeddings = self.edge_embedding(edge_attr.float())
        return edge_embeddings

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.get_edge_embedding(edge_attr)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0,
                 atom_feat_dim=None, bond_feat_dim=None):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.name = 'gin'

        self.pool = global_mean_pool

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if atom_feat_dim is None:
            self.basic_feat = True
            self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
            self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
            torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        else:
            self.basic_feat = False
            self.x_embedding = nn.Linear(atom_feat_dim, emb_dim)
            torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)

        # List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINConv(emb_dim, aggr="add", edge_feat_dim=bond_feat_dim))

        # List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))


    def forward(self, x, edge_index, edge_attr, batch, x_feat=None):
        if self.basic_feat:
            x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        else:
            x = self.x_embedding(x.float())

        if x_feat is not None:
            batch_x, batch_mask = to_dense_batch(x, batch)
            x_feat = self.additional_lin(x_feat.view(batch_x.size(0), -1))
            batch_x[(torch.arange(batch_x.size(0)), batch_mask.sum(-1)-1)] += x_feat
            x = batch_x[batch_mask]

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
            graph_representation = self.pool(node_representation, batch)
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)
            graph_representation = self.pool(node_representation, batch)

        return graph_representation, node_representation


class GPS(torch.nn.Module):
    def __init__(self, channels: int, pe_dim: int, node_dim: int, edge_dim: int, heads: int,
                 num_layers: int, attn_type: str, dropout: float, attn_dropout: float):
        super().__init__()

        self.name = 'gps'
        self.node_emb = nn.Linear(node_dim, channels - pe_dim)
        self.pe_lin = nn.Linear(20, pe_dim)
        self.pe_norm = nn.BatchNorm1d(20)
        self.edge_emb = nn.Linear(edge_dim, channels)
        self.dropout = dropout
        self.attn_dropout = attn_dropout

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(channels, channels),
                nn.ReLU(),
                nn.Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(layer), heads=heads, dropout=self.dropout,
                           attn_type=attn_type, attn_kwargs={'dropout': self.attn_dropout})
            self.convs.append(conv)

        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x, edge_attr = x.float(), edge_attr.float()
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, batch, edge_attr=edge_attr)

        out = global_add_pool(x, batch)
        return out, x


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1
