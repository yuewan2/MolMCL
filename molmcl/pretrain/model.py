import math
import copy
import numpy as np
from itertools import combinations, product
from collections import defaultdict
from rdkit import DataStructs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch.nn.utils.rnn import pad_sequence

from molmcl.models.aggr import PromptAggr


class GNNWrapper(nn.Module):
    def __init__(self, gnn, emb_dim, config, atom_context_size, aggr_head=4, layer_norm=True, layer_norm_out=True):
        super(GNNWrapper, self).__init__()

        self.device = config['device']
        self.margin = config['optim']['margin']
        self.adaptive_margin_coeff = config['optim']['adamg_coeff']
        self.dist_metric = config['optim']['distance_metric']
        self.num_candidates = config['optim']['num_candidates']
        self.dropout_ratio = config['model']['dropout_ratio']
        self.knowledge_guided = config['optim']['knowledge_guided']

        # hard-coded prompts
        self.prompt_token = ['<molsim>', '<scaffsim>', '<context>']
        self.prompt_inds = torch.LongTensor(
            [0, 1, 2] + [0] * (self.num_candidates - 1) + [1] * (self.num_candidates - 1) + [2])

        # Main GNN and aggregation modules:
        self.emb_dim = emb_dim
        self.gnn = gnn
        self.aggrs = nn.ModuleList([PromptAggr(emb_dim=emb_dim,
                                               num_heads=aggr_head,
                                               dropout=self.dropout_ratio,
                                               local_loss=i==(len(self.prompt_token)-1),
                                               layer_norm_out=layer_norm_out)
                                    for i in range(len(self.prompt_token))])
        # Other prediction heads:
        self.context_func = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, atom_context_size)
        )
        self.motif_func = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, 86)
        )
        if self.knowledge_guided:
            self.fp_func = nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.ReLU(),
                nn.Linear(emb_dim, 512)
            )
            self.md_func = nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.ReLU(),
                nn.Linear(emb_dim, 114)
            )
            

    def distance(self, tensor_a, tensor_b, metric=None):
        if (metric is None and self.dist_metric == 'cossim') or metric == 'cossim':
            return 1 - F.cosine_similarity(tensor_a, tensor_b, dim=-1)
        elif (metric is None and self.dist_metric == 'l2norm') or metric == 'l2norm':
            return (tensor_a - tensor_b).norm(dim=-1)
        else:
            raise Exception

    
    def get_representation(self, batch):
        # Graph encoder:
        if self.gnn.name == 'gps':
            _, h_node = self.gnn(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
        else:
            _, h_node = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Prompt guided aggregation:
        batch_x, batch_mask = to_dense_batch(h_node, batch.batch)
        batch_size = batch_x.size(0) // (self.num_candidates * 2 + 2)
        prompt_inds = torch.concat([self.prompt_inds] * batch_size).to(batch_x.device)

        graph_reps = []
        for i in range(len(self.prompt_token)):
            graph_reps.append(
                self.aggrs[i](batch_x[prompt_inds == i], batch_mask[prompt_inds == i]),
            )
        return graph_reps

    def forward(self, batch):
        graph_rep = self.get_representation(batch)
        output = self.compute_loss(graph_rep, batch)
        output['graph_rep'] = graph_rep
        return output

    def compute_loss(self, graph_rep, batch):
        device = graph_rep[0][0].device
        batch_size = (len(batch) // (2 * self.num_candidates + 2))

        # First loss:
        mol_fps = [batch.mol_fps[i][0] for i in range(len(batch.mol_fps)) if i % (self.num_candidates * 2 + 2) == 0]
        mol_dist = 1 - torch.Tensor(
            [DataStructs.BulkTanimotoSimilarity(mol_fps[i], mol_fps) for i in range(len(mol_fps))]).to(device)

        h_gr = graph_rep[0][0]
        h_gr = h_gr.view(batch_size, self.num_candidates, self.emb_dim)
        loss_1 = self.adaptive_margin_loss(F.normalize(h_gr, dim=-1), mol_dist)

        aggr_score = graph_rep[0][2]
        aggr_score = aggr_score.view(batch_size, self.num_candidates, -1)
        aggr_score_regu = aggr_score.clone().detach()
        aggr_score_regu[:] = (1 / (aggr_score > 0).sum(dim=-1, keepdim=True))
        aggr_score_regu[aggr_score == 0] = 0
        loss_1_reg = F.smooth_l1_loss(
            aggr_score.reshape(-1, aggr_score.size(-1)),
            aggr_score_regu.reshape(-1, aggr_score_regu.size(-1)))

        # Second loss:
        scaff_fps = [batch.scaff_fps[i][0] for i in range(len(batch.scaff_fps)) if i % (self.num_candidates * 2 + 2) == 0]
        scaff_dist = 1 - torch.Tensor(
            [DataStructs.BulkTanimotoSimilarity(scaff_fps[i], scaff_fps) for i in range(len(scaff_fps))])
        scaff_dist = scaff_dist.to(device)

        h_gr = graph_rep[1][0]
        h_gr = h_gr.view(batch_size, self.num_candidates, self.emb_dim)
        loss_2 = self.adaptive_margin_loss(F.normalize(h_gr, dim=-1), scaff_dist)

        prompt_inds = torch.concat([self.prompt_inds] * batch_size)
        regu_inds = [batch.regu_inds[i] for i in range(len(batch.regu_inds)) if (prompt_inds == 1)[i]]
        aggr_score = graph_rep[1][2].squeeze(1)
        aggr_score_regu = torch.zeros(aggr_score.size()).to(aggr_score.device)
        # aggr_score.new_tensor(aggr_score.data).fill_(0)
        for i in range(aggr_score_regu.size(0)):
            if len(regu_inds[i]):
                aggr_score_regu[i][regu_inds[i]] = 1 / len(regu_inds[i])
        loss_2_reg = F.smooth_l1_loss(
            aggr_score.reshape(-1, aggr_score.size(-1)),
            aggr_score_regu.reshape(-1, aggr_score_regu.size(-1)))

        # Third loss:
        context_labels = [torch.LongTensor(batch.context_label[i]) for i in range(len(batch.context_label)) if
                          i % (self.num_candidates * 2 + 2) == (self.num_candidates * 2 + 2) - 1]
        context_labels = pad_sequence(context_labels, batch_first=True, padding_value=0).to(device)
        h_tk = graph_rep[2][1]
        h_tk = h_tk.view(batch_size, 2, -1, self.emb_dim)
        h_tk = h_tk[:, -1]  # last sample
        h_tk = h_tk[:, :context_labels.size(1)].contiguous().view(-1, self.emb_dim)
        context_logits = self.context_func(h_tk)
        context_loss = F.cross_entropy(context_logits, context_labels.view(-1), reduction='none')
        context_loss = (context_loss[context_labels.view(-1) != 0]).mean()

        motif_labels = torch.FloatTensor(
            np.array([np.array(batch.mol_mds[i][0][-86:]) for i in range(len(batch.mol_mds)) if
             i % (self.num_candidates * 2 + 2) == 0])).to(
            h_gr.device)
        h_gr = graph_rep[2][0]
        h_gr = h_gr.view(batch_size, 2, self.emb_dim)
        h_gr = h_gr[:, 0]  
        motif_scores = self.motif_func(h_gr)
        motif_loss = F.smooth_l1_loss(motif_scores, motif_labels.float())

        loss_3 = (context_loss + motif_loss) / 2

        losses = {'loss_1': loss_1, 'loss_2': loss_2, 'loss_3': loss_3, 'loss_1_reg': loss_1_reg, 'loss_2_reg': loss_2_reg}

        # Fourth loss:
        if self.knowledge_guided:
            graph_value = torch.stack(
                [graph_rep[i][0].view(batch_size, -1, self.emb_dim)[:, 0] for i in range(len(graph_rep))]).transpose(0, 1)

            # Preset #1 
            preset_weight_1 = torch.FloatTensor([0.45, 0.1, 0.45]).to(graph_value.device)
            graph_rep_1 = torch.matmul(graph_value.transpose(1, 2), preset_weight_1.unsqueeze(1)).squeeze(2)

            mol_mds = torch.from_numpy(np.array([np.array(batch.mol_mds[i][0][:-86]) for i in range(len(batch.mol_mds)) if
                i % (self.num_candidates * 2 + 2) == 0])).to(graph_value.device)
            md_scores = self.md_func(graph_rep_1)
            md_loss = F.smooth_l1_loss(md_scores[~torch.isnan(mol_mds)], mol_mds[~torch.isnan(mol_mds)])

            # Preset #2
            preset_weight_2 = torch.FloatTensor([0.1, 0.45, 0.45]).to(graph_value.device)
            graph_rep_2 = torch.matmul(graph_value.transpose(1, 2), preset_weight_2.unsqueeze(1)).squeeze(2)

            scaff_mds = torch.from_numpy(np.array([np.array(batch.scaff_mds[i][0][:-86]) for i in range(len(batch.scaff_mds)) if
                i % (self.num_candidates * 2 + 2) == 0])).to(graph_value.device)
            scaff_md_scores = self.md_func(graph_rep_2)
            scaff_md_loss = F.smooth_l1_loss(scaff_md_scores[~torch.isnan(scaff_mds)], scaff_mds[~torch.isnan(scaff_mds)])

            # Preset #3
            preset_weight_3 = torch.FloatTensor([0.3, 0.1, 0.6]).to(graph_value.device)
            graph_rep_3 = torch.matmul(graph_value.transpose(1, 2), preset_weight_3.unsqueeze(1)).squeeze(2)

            mol_fps = torch.FloatTensor(
                np.array([np.array(batch.mol_fps[i][0]) for i in range(len(batch.mol_fps)) if 
                 i % (self.num_candidates * 2 + 2) == 0])).to(graph_value.device)
            fp_scores = self.fp_func(graph_rep_3)
            fp_loss = F.binary_cross_entropy_with_logits(fp_scores.view(-1), mol_fps.view(-1))

            losses['loss_4'] = md_loss + fp_loss

        return losses

    def margin_loss(self, graph_rep, margin_factor=None, anchor_inds=None):
        num_candidates = self.num_candidates
        batch_size = graph_rep.size(0)

        # compute positive sim
        if anchor_inds is not None:
            sample_mask = torch.zeros((batch_size, num_candidates)).bool()
            sample_mask[(torch.arange(batch_size), anchor_inds)] = True
            anchor = graph_rep[sample_mask].unsqueeze(1)
            pos = graph_rep[~sample_mask].view(batch_size, num_candidates - 1, -1)
            graph_rep = torch.concat([anchor, pos], dim=1)

        anchor = graph_rep[:, :1]
        pos = graph_rep[:, 1:]

        pos_sim = self.distance(anchor, pos)
        # print(pos_sim)
        pos_sim = pos_sim.repeat_interleave(batch_size - 1, dim=0).view(batch_size, batch_size - 1,
                                                                        -1)  # pos_sim : (B, B-1, num_candidates-1)

        # Compute negative sim
        ori_rep = graph_rep[:, 0]

        interleave = ori_rep.repeat_interleave(batch_size, dim=0).view(batch_size, batch_size, -1)
        repeat = ori_rep.repeat(batch_size, 1).view(batch_size, batch_size, -1)
        sim_matrix = self.distance(interleave, repeat)  # sim_matrix : (B, B)

        # Remove diagonal of ''sim_matrix''
        diag_idx = torch.eye(batch_size, dtype=torch.bool)
        sim_matrix = sim_matrix[~diag_idx].view(batch_size, batch_size - 1)  # sim_matrix: (B, B-1)

        # neg_sim: (B, B-1, num_candidates-1)
        neg_sim = sim_matrix.repeat_interleave(num_candidates - 1, dim=-1).view(batch_size, batch_size - 1,
                                                                                num_candidates - 1)

        # Loss computation
        if margin_factor is not None:
            # margin_factor: (B, B)
            diag_idx = torch.eye(batch_size, dtype=torch.bool)
            # margin_factor: (B, B-1)
            margin_factor = margin_factor[~diag_idx].view(batch_size, batch_size - 1)
            margin_factor_edge = torch.ones((batch_size, batch_size - 1, num_candidates - 1),
                                            device=margin_factor.device)
            margin_factor_edge[margin_factor == 0] = 0
            # margin_factor: (B, B-1, num_candidates-1)
            margin_factor = margin_factor.repeat_interleave(num_candidates - 1, dim=-1).view(batch_size, batch_size - 1,
                                                                                             num_candidates - 1)
            margin = margin_factor * self.margin
        else:
            margin = self.margin
            margin_factor_edge = torch.ones((batch_size, batch_size - 1, num_candidates - 1),
                                            device=graph_rep.device)

        margin_loss = torch.maximum(torch.tensor(0).to(self.device),
                                    margin_factor_edge * (margin + pos_sim - neg_sim)).mean()

        return margin_loss

    def adaptive_margin_loss(self, graph_rep, scaff_dist, anchor_inds=None):
        batch_size = graph_rep.size(0)
        num_candidates = graph_rep.size(1)

        if anchor_inds is not None:
            # Re-allocate anchor and positive samples
            sample_mask = torch.zeros((batch_size, num_candidates)).bool()
            sample_mask[(torch.arange(batch_size), anchor_inds)] = True
            anchor = graph_rep[sample_mask].unsqueeze(1)
            pos = graph_rep[~sample_mask].view(batch_size, num_candidates - 1, -1)
            graph_rep = torch.concat([anchor, pos], dim=1)

        margin_loss = self.margin_loss(graph_rep, margin_factor=scaff_dist)

        # graph_rep: (B, num_candidate, emb_dim)
        batch_size = graph_rep.size(0)
        scaff_dist_diff = scaff_dist.unsqueeze(1) - scaff_dist.unsqueeze(2)  # scaff_dist_diff: (B, B, B)
        # indexing of scaff_dist_diff[0, 1, 2] represents: dist(g_2, g_0) - dist(g_1, g_0)

        inds_triplet = torch.argwhere(scaff_dist_diff > 0.3)
        inds_triplet = inds_triplet[(inds_triplet[:, 0] != inds_triplet[:, 1]) &
                                    (inds_triplet[:, 0] != inds_triplet[:, 2])]

        diff_values = scaff_dist_diff[inds_triplet[:, 0], inds_triplet[:, 1], inds_triplet[:, 2]]
        if len(diff_values):
            if len(diff_values) > batch_size ** 2:
                sub_inds = torch.argsort(diff_values, descending=True)[:batch_size ** 2]
                diff_values = diff_values[sub_inds]
                inds_triplet = inds_triplet[sub_inds]

            anchor_inds, pos_inds, neg_inds = inds_triplet[:, 0], inds_triplet[:, 1], inds_triplet[:, 2]
            anchor_rep, pos_rep, neg_rep = graph_rep[anchor_inds, 0], graph_rep[pos_inds, 0], graph_rep[neg_inds, 0]

            pos_sim = self.distance(anchor_rep, pos_rep)
            neg_sim = self.distance(anchor_rep, neg_rep)

            margin_loss_adaptive = torch.maximum(torch.tensor(0).to(graph_rep.device),
                                                 diff_values.detach() * self.margin + pos_sim - neg_sim).mean()

            return self.adaptive_margin_coeff * margin_loss_adaptive + margin_loss
        else:
            return margin_loss

