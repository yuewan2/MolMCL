import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import softmax
from torch_geometric.nn.aggr import AttentionalAggregation


class PromptAggr(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.0, local_loss=False, layer_norm_out=True):
        super(PromptAggr, self).__init__()

        assert emb_dim % num_heads == 0
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.local_loss = local_loss
        self.attn = nn.MultiheadAttention(
            emb_dim,
            num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.prompt_query = nn.Parameter(torch.FloatTensor(emb_dim))
        nn.init.normal_(self.prompt_query)

        self.linear = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.layer_norm_1 = nn.LayerNorm(emb_dim) if layer_norm_out else None
        if local_loss:
            self.layer_norm_2 = nn.LayerNorm(emb_dim) if layer_norm_out else None
            

    def forward(self, batch_x, batch_mask, score_head_i=0):
        batch_size = batch_x.size(0)
        batch_x = self.linear(batch_x)
        global_out, score = self.attn(self.prompt_query.expand(batch_size, 1, self.emb_dim),
                                      batch_x, batch_x, ~batch_mask, average_attn_weights=False)
        global_out = global_out.squeeze(1)

        if self.layer_norm_1 is not None:
            global_out = self.layer_norm_1(global_out)

        if self.local_loss:
            if self.layer_norm_2 is not None:
                local_out = self.layer_norm_2(batch_x)
            else:
                local_out = batch_x
        else:
            local_out = None

        if score_head_i > -1:
            score = score[:, score_head_i]

        return global_out, local_out, score