import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.checkpoint import checkpoint
from copy import deepcopy


def mean_normalize(x: torch.Tensor, mask: torch.Tensor):
    valid_mask = (~mask).unsqueeze(-1).float()
    sum_embeddings = (x * valid_mask).sum(dim=1)  # (bsz, dim)
    lengths = valid_mask.sum(dim=1)  # (bsz, 1)
    mean_embeddings = sum_embeddings / lengths.clamp(min=1e-9)
    mean_embeddings = F.normalize(mean_embeddings, dim=-1)
    return mean_embeddings


class clshead(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.use_proj = config.use_proj
        if config.use_proj:
            self.model = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(input_dim, config.d_model),
            )

    def forward(self, emb, *args, **kwargs):
        if self.use_proj:
            emb = self.model(emb)
        return F.normalize(emb[:, 0], dim=-1)


class ProjHead(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )

    def forward(self, x, *args, **kwargs):
        x = self.model(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(input_dim, config.d_model),
        )

    def forward(self, x, *args, **kwargs):
        x = self.model(x)
        return x


class TransformerHead(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.proj_first = config.proj_first
        if self.proj_first:
            transformer_dim = config.d_model
        else:
            transformer_dim = input_dim
        dim_feedforward = config.dim_feedforward if hasattr(config, "dim_feedforward") else 4*transformer_dim
        
        self.proj = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=transformer_dim,
            nhead=config.nhead,
            dim_feedforward=dim_feedforward,
            dropout=config.dropout
        )
        self.layers = nn.ModuleList(
            [encoder_layer for _ in range(config.num_layers)]
        )
    
    def forward(self, x, pad_mask, *args, **kwargs):
        if self.proj_first:
            x = self.proj(x)
        for layer in self.layers:
            x = checkpoint(layer, x, None, pad_mask, use_reentrant=False)
        if not self.proj_first:
            x = self.proj(x)
        return x
    

class TaskHead(nn.Module):
    def __init__(self, config, input_dim, bihead):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.input_dim = input_dim
        self.bihead = bihead
        if self.model_type == 'transformer':
            self.head = TransformerHead(config, input_dim)
        elif self.model_type == 'proj_only':
            self.head = ProjHead(config, input_dim)
        elif self.model_type == 'mlp':
            self.head = MLPHead(config, input_dim)
        elif self.model_type == 'cls':
            self.head = clshead(config, input_dim)
        else:
            raise NotImplementedError
        if self.bihead:
            self.head2 = deepcopy(self.head)
    
    def forward(self, emb, pad_mask, attn=None):
        out = self.head(emb, pad_mask, attn)
        if out.dim() == 3:
            out = mean_normalize(out, pad_mask)
        return out