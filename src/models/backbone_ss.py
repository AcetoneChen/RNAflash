import numpy as np
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt
from torch.optim import Adam, AdamW
from lightning import LightningModule

from .backbone import Backbone
from ..utils import ModelConfig


# ==============================================================================
# SS label definition
# ==============================================================================

def dotbracket_to_contact(ss: str) -> np.ndarray:
    L = len(ss)
    contact = np.zeros((L, L), dtype=np.float32)
    open_chars = {'(', '[', '{', '<'}
    close_to_open = {')': '(', ']': '[', '}': '{', '>': '<'}
    stacks = {c: [] for c in open_chars}
    for i, ch in enumerate(ss):
        if ch in open_chars:
            stacks[ch].append(i)
        elif ch in close_to_open:
            open_ch = close_to_open[ch]
            if stacks[open_ch]:
                j = stacks[open_ch].pop()
                contact[j, i] = 1.0
                contact[i, j] = 1.0
    return contact


class SSResBlock(nn.Module):
    """Residual block used by the ERNIE-RNA SS head.

    Shape-compatible with the checkpoint's ``proj.resnet.{i}``:
        bn1   : BatchNorm2d(inplanes)
        conv1 : Conv2d(inplanes, planes, 3, padding=1, bias=False)
        conv2 : Conv2d(planes, planes, 3, stride, padding=dilation,
                       bias=False, dilation=dilation)
    """

    def __init__(self, inplanes=64, planes=64, stride=1, groups=1,
                 base_width=64, dilation=1):
        super().__init__()
        assert groups == 1 and base_width == 64
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=0.3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False,
                               dilation=dilation)
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.relu2(out)
        out = self.conv2(out)
        return out + identity


def _proj_forward(proj: nn.Sequential, x: torch.Tensor, use_ckpt: bool) -> torch.Tensor:
    """Forward through ``proj`` (resnet blocks + final conv), checkpointing each
    res block individually. Wrapping the whole proj as ONE checkpoint segment would
    re-materialize every block's activations at once (no memory saved); per-block
    segments keep only one block's activations live at a time."""
    if not use_ckpt:
        return proj(x)
    for blk in proj.resnet:
        x = ckpt(blk, x, use_reentrant=False)
    return proj.final(x)


def _slice_attn_to_valid(attn_map: torch.Tensor, mask: torch.Tensor,
                         head_idx: int = None) -> torch.Tensor:
    """Slice the last-layer attention map to the real RNA region, per sample.

    Official ERNIE-RNA uses ``attn_map[-1][:, h, 1:-1, 1:-1]``, which assumes EOS
    sits at the last token position (i.e. no padding in the batch). With batched
    padding the token layout is ``[CLS(0), seq(1..L), EOS(L+1), PAD(L+2..)]``, so
    EOS is *not* at -1. We therefore slice each sample to its own real length and
    re-pad to the batch's max real length — reproducing the official "head only
    ever sees real nucleotides" behaviour.

    Args:
        attn_map: [B, num_heads, T, T] accumulated attention logits (incl. CLS/EOS).
        mask: [B, T] bool, True = padding. If None, treated as no padding.
        head_idx: attention head index to keep. If None, keeps all heads.

    Returns:
        [B, 1, max_l, max_l] (single head) or [B, num_heads, max_l, max_l] (all heads).
    """
    if head_idx is not None:
        attn_map = attn_map[:, head_idx:head_idx + 1]     # [B, 1, T, T]
    if mask is None:
        return attn_map[:, :, 1:-1, 1:-1]
    lens = (~mask).sum(dim=1) - 2                 # real seq len (strip CLS, EOS)
    max_l = int(lens.max())
    slices = []
    for b in range(attn_map.size(0)):
        l = int(lens[b])
        m = attn_map[b, :, 1:l + 1, 1:l + 1]      # [H, l, l], real positions 1..L
        slices.append(F.pad(m, (0, max_l - l, 0, max_l - l)))
    return torch.stack(slices)                    # [B, H, max_l, max_l]


class AttentionSSHead(nn.Module):
    def __init__(self, num_heads: int = 12, attn_head_idx: int = 5, depth: int = 8,
                 use_ckpt: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.attn_head_idx = attn_head_idx
        self.use_ckpt = use_ckpt

        self.conv1 = nn.Conv2d(1, 8, 7, 1, 3)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.conv2 = nn.Conv2d(8, 63, 7, 1, 3)

        res_layers = []
        for i in range(depth):
            dilation = pow(2, (i % 3))
            res_layers.append(SSResBlock(inplanes=64, planes=64, dilation=dilation))
        res_layers = nn.Sequential(*res_layers)

        layers = OrderedDict()
        layers["resnet"] = res_layers
        layers["final"] = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.proj = nn.Sequential(layers)

    def forward(self, attn_map: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        final_attn = _slice_attn_to_valid(attn_map, mask, self.attn_head_idx)
        final_attn = torch.nan_to_num(final_attn, nan=0.0, posinf=0.0, neginf=0.0)
        out = self.conv1(final_attn)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = torch.cat((out, final_attn), dim=1)   # 63 + 1 = 64 channels
        out = _proj_forward(self.proj, out, self.use_ckpt)   # per-res-block ckpt saves memory
        return (out + out.permute(0, 1, 3, 2))      # symmetrize -> [B, 1, max_l, max_l]


class TinyAttentionSSHead(nn.Module):
    def __init__(self, num_heads: int = 12, attn_head_idx: int = 5):
        super().__init__()
        self.num_heads = num_heads
        self.attn_head_idx = attn_head_idx

        self.conv1 = nn.Conv2d(1, 8, 7, 1, 3)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.conv2 = nn.Conv2d(8, 32, 5, 1, 2)
        self.relu2 = nn.ReLU(inplace=True)
        # 32 (conv2) + 1 (final_attn skip) -> 1
        self.conv3 = nn.Conv2d(33, 1, 3, 1, 1)

    def forward(self, attn_map: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        final_attn = _slice_attn_to_valid(attn_map, mask, self.attn_head_idx)
        final_attn = torch.nan_to_num(final_attn, nan=0.0, posinf=0.0, neginf=0.0)
        out = self.relu(self.conv1(final_attn))
        out = self.dropout(out)
        out = self.relu2(self.conv2(out))
        out = torch.cat((out, final_attn), dim=1)   # [B, 33, L, L]
        out = self.conv3(out)
        return (out + out.permute(0, 1, 3, 2))      # symmetrize -> [B, 1, max_l, max_l]


class MLPAttnHead(nn.Module):
    def __init__(self, num_heads: int = 12, hidden: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(num_heads, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, attn_map: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn = _slice_attn_to_valid(attn_map, mask, head_idx=None)  # [B, H, L, L]
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        x = attn.permute(0, 2, 3, 1)                # [B, L, L, H]
        logits = self.mlp(x).squeeze(-1)            # [B, L, L]
        return ((logits + logits.transpose(1, 2)) / 2).unsqueeze(1)


class CNN12Head(nn.Module):
    def __init__(self, num_heads: int = 12, depth: int = 8, use_ckpt: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.use_ckpt = use_ckpt
        self.conv1 = nn.Conv2d(num_heads, 8, 7, 1, 3)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.conv2 = nn.Conv2d(8, 64 - num_heads, 7, 1, 3)   # 52 for 12 heads

        res_layers = []
        for i in range(depth):
            dilation = pow(2, (i % 3))
            res_layers.append(SSResBlock(inplanes=64, planes=64, dilation=dilation))
        res_layers = nn.Sequential(*res_layers)

        layers = OrderedDict()
        layers["resnet"] = res_layers
        layers["final"] = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.proj = nn.Sequential(layers)

    def forward(self, attn_map: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        final_attn = _slice_attn_to_valid(attn_map, mask, head_idx=None)  # [B, H, L, L]
        final_attn = torch.nan_to_num(final_attn, nan=0.0, posinf=0.0, neginf=0.0)
        out = self.conv1(final_attn)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = torch.cat((out, final_attn), dim=1)   # (64-H) + H = 64
        out = _proj_forward(self.proj, out, self.use_ckpt)   # per-res-block ckpt saves memory
        return (out + out.permute(0, 1, 3, 2))      # symmetrize -> [B, 1, max_l, max_l]
    

class DirectContact(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, emb, mask):
        emb = self.mlp(emb)
        contact = torch.bmm(emb, emb.transpose(1, 2)).unsqueeze(1)
        contact = _slice_attn_to_valid(contact, mask).squeeze(1)
        L = contact.shape[1]
        diag_mask = torch.diag(torch.ones(L, dtype=bool, device=contact.device))
        contact = contact.masked_fill(diag_mask, -1e8)
        return contact



# ==============================================================================
# Fine-tuning model
# ==============================================================================

class SSModel(LightningModule):
    def __init__(self, config, DP=False, *args, **kwargs):
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        self.save_hyperparameters()
        self.config = config
        self.DP = DP

        if config.pretrained_model != 'ernie_rna':
            raise NotImplementedError(
                f'SSFTModel currently only supports ernie_rna backbone, '
                f'got {config.pretrained_model}.'
            )

        self.backbone = Backbone(
            config.pretrained_model,
            bibert=getattr(config, 'bibert', False),
            freeze=getattr(config, 'freezebert', False)
        )
        self.head = self._build_head()


    def _build_head(self) -> nn.Module:
        head_type = getattr(self.config, 'head_type', 'attn')
        use_ckpt = getattr(self.config, 'head_ckpt', True)
        if head_type == 'full_attn':
            return AttentionSSHead(use_ckpt=use_ckpt)
        elif head_type == 'tiny_attn':
            return TinyAttentionSSHead()
        elif head_type == 'mlp_attn':
            return MLPAttnHead(num_heads=12)
        elif head_type == 'cnn12':
            return CNN12Head(num_heads=12, use_ckpt=use_ckpt)
        elif head_type == 'contact':
            return DirectContact(self.backbone.bert_dim, self.backbone.bert_dim, 256)
        raise NotImplementedError(f'Unknown head_type: {head_type}')


    def forward(self, toks: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return contact-map logits ``[B, 1, max_l, max_l]`` (real RNA region only)."""
        _emb, attn_map = self.backbone(toks, mask, get_attn=True)
        if isinstance(self.head, DirectContact):
            logits = self.head(_emb, mask)
        else:
            logits = self.head(attn_map, mask)
        L = logits.shape[-1]
        diag_mask = torch.diag(torch.ones(L, dtype=bool, device=logits.device))
        logits = logits.masked_fill(diag_mask, -1e8)
        return logits


    def load_attn_map_ss_pretrained(self, ckpt_path: str):
        """Load backbone + SS head from an ERNIE-RNA attn-map SS checkpoint.

        E.g. ERNIE-RNA_attn-map_ss_prediction_bpRNA-1m-all_and_RNAStralign_checkpoint.pt:
            module.sentence_encoder.* -> backbone, module.conv1/conv2/proj.* -> head.
        """
        data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        enc_sd = {k.replace('module.sentence_encoder', 'encoder'): v
                  for k, v in data.items() if k.startswith('module.sentence_encoder')}
        missing, unexpected = self.backbone.bert.load_state_dict(enc_sd, strict=False)
        if missing:
            print(f'[SSFTModel] backbone missing keys (expected): {missing}')
        if unexpected:
            print(f'[SSFTModel] backbone unexpected keys: {unexpected}')
        head_sd = {k.replace('module.', ''): v for k, v in data.items()
                   if k.startswith('module.conv') or k.startswith('module.proj')}
        missing_h, unexpected_h = self.head.load_state_dict(head_sd, strict=True)
        assert not missing_h and not unexpected_h, (missing_h, unexpected_h)
        print('[SSFTModel] SS head loaded with strict=True (all params matched).')
