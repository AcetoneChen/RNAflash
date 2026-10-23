"""
use Pure cross attention instead
All-param finetune induces overfitting (?)
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.checkpoint import checkpoint
from torch.nn import BCEWithLogitsLoss
from lightning import LightningModule
import numpy as np
from numba import njit

from .backbone_ss import SSModel
from ..utils import ModelConfig, config_to_dict


@njit
def decode_numba(S_np):
    Lx, Ly = S_np.shape
    dp = np.zeros((Lx + 1, Ly + 1))
    trace = np.zeros((Lx + 1, Ly + 1), dtype=np.int32)

    for i in range(1, Lx + 1):
        for j in range(1, Ly + 1):
            up = dp[i - 1, j]
            left = dp[i, j - 1]
            diag = dp[i - 1, j - 1] + S_np[i - 1, j - 1]

            if up >= left and up >= diag:
                dp[i, j] = up
                trace[i, j] = 0   # up
            elif left >= up and left >= diag:
                dp[i, j] = left
                trace[i, j] = 1   # left
            else:
                dp[i, j] = diag
                trace[i, j] = 2   # left up

    # traceback
    pairs = []
    i, j = Lx, Ly
    while i > 0 and j > 0:
        if trace[i, j] == 2:
            i -= 1
            j -= 1
            pairs.append((i, j))
        elif trace[i, j] == 1:
            j -= 1
        else:
            i -= 1
    pairs.reverse()
    return pairs

def decode_optimal_pairs(S):
    # torch.Tensor to numpy
    S_np = S.detach().cpu().to(torch.float32).numpy()
    return decode_numba(S_np)


def build_alignment(seq1, seq2, pairs):
    aln1 = []
    aln2 = []
    i = j = 0
    for pi, pj in pairs:
        while i < pi:
            aln1.append(seq1[i])
            aln2.append("-")
            i += 1
        while j < pj:
            aln1.append("-")
            aln2.append(seq2[j])
            j += 1
        aln1.append(seq1[pi])
        aln2.append(seq2[pj])

        i = pi + 1
        j = pj + 1

    while i < len(seq1):
        aln1.append(seq1[i])
        aln2.append("-")
        i += 1

    while j < len(seq2):
        aln1.append("-")
        aln2.append(seq2[j])
        j += 1

    return "".join(aln1), "".join(aln2)

def apply_remove_query_gap(query_gap:str, target_gap:str, gap_char='-'):
    new_query_gap = query_gap.replace(gap_char, '')
    keep_mask = [c != gap_char for c in query_gap]
    new_target_gap = ''.join(target_gap[i] for i, keep in enumerate(keep_mask) if keep)
    return new_query_gap, new_target_gap


def remove_cls_eos_from_mask(original_mask:torch.Tensor):
    bsz, seq_len = original_mask.shape
    device = original_mask.device
    new_mask = original_mask.clone()
    is_valid = ~original_mask  # False=pad, True=valid
    seq_lens = is_valid.sum(dim=1)  # (bsz,)
    new_mask[:, 0] = True
    eos_positions = seq_lens - 1
    eos_positions = torch.clamp(eos_positions, min=0)
    new_mask[torch.arange(bsz, device=device), eos_positions] = True
    return new_mask


class CrossTransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, q_mask, kv_mask):
        self_attn_out, _ = self.self_attn(query=q, key=q, value=q,
                                      key_padding_mask=q_mask, need_weights=False)
        q = self.norm1(q + self_attn_out)
        cross_attn_out, _ = self.cross_attn(query=q, key=kv, value=kv,
                                      key_padding_mask=kv_mask, need_weights=False)
        q = self.norm2(q + cross_attn_out)
        ffn_out = self.ffn(q)
        q = self.norm3(q + ffn_out)
        return q


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.h = num_heads
        self.dh = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
    
    def forward(self, x, y):
        _, M, _ = x.shape
        B, N, _ = y.shape
        h = self.h
        dh = self.dh

        Q = self.q_proj(x)  # (B, M, dim)
        K = self.k_proj(y)  # (B, N, dim)

        Q = Q.view(B, M, h, dh).transpose(1, 2)  # (B, h, M, dh)
        K = K.view(B, N, h, dh).transpose(1, 2)  # (B, h, N, dh)

        attn = (Q @ K.transpose(-2, -1)) / (dh ** 0.5)  # (B, h, M, N)
        return attn.permute(0, 2, 3, 1)


class CrossFeatExtractor(nn.Module):
    def __init__(self, input_dim, dim=256, num_layers=3, num_heads=8) -> None:
        super().__init__()
        self.dim = dim
        self.in_proj = nn.Linear(input_dim, dim)
        self.in_emb = nn.ModuleList(
            [CrossTransformerLayer(self.dim, num_heads) for _ in range(num_layers)]
        )
        self.cross_module = CrossAttention(dim, num_heads)
        self.ali_proj = nn.Sequential(
            nn.Linear(num_heads, num_heads*4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_heads*4, 1)
        )
        self.local_rerank_proj = nn.Sequential(
            nn.Linear(num_heads, num_heads*4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_heads*4, 1)
        )
        self.global_rerank_proj = nn.Sequential(
            nn.Linear(dim*2, dim*4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim*4, 1)
        )

    def get_cross_feat(self, q_emb, q_mask, c_emb, c_mask):
        q_emb = self.in_proj(q_emb)
        c_emb = self.in_proj(c_emb)
        for layer in self.in_emb:
            new_q_emb = checkpoint(layer, q_emb, c_emb, q_mask, c_mask, use_reentrant=False)
            new_c_emb = checkpoint(layer, c_emb, q_emb, c_mask, q_mask, use_reentrant=False)
            q_emb = new_q_emb
            c_emb = new_c_emb
        cross_feat = checkpoint(self.cross_module, q_emb, c_emb, use_reentrant=False)
        return cross_feat, q_emb, c_emb
    
    def get_local_logits(self, cross_feat, q_valid, c_valid, cross_mask):
        local_reranking_logits = self.local_rerank_proj(cross_feat).squeeze(-1)  # (B, M+2, N+2)

        B = local_reranking_logits.shape[0]
        valid_q_lens = q_valid.sum(dim=1)  # (B,)
        valid_c_lens = c_valid.sum(dim=1)  # (B,)
        K = torch.clamp(torch.min(valid_q_lens, valid_c_lens), min=1)  # (B,)

        # Mask invalid positions, then select top-K per sample
        masked_logits = local_reranking_logits.masked_fill(~cross_mask, -1e7) # cls and eos are set to -inf
        flat_logits = masked_logits.view(B, -1)  # (B, (M+2)*(N+2))
        max_K = K.max().item()
        topk_vals, _ = torch.topk(flat_logits, k=max_K, dim=-1)  # (B, max_K)

        # Per-sample mask: keep only first K_i values
        k_mask = torch.arange(max_K, device=local_reranking_logits.device).unsqueeze(0) < K.unsqueeze(1)
        local_reranking_logits = (topk_vals * k_mask).sum(dim=1) / K.float()  # (B,)
        return local_reranking_logits

    def forward(self, q_emb, q_mask, c_emb, c_mask):
        cross_feat, q_emb, c_emb = self.get_cross_feat(
            q_emb, q_mask, c_emb, c_mask
        )

        pairing_logits = self.ali_proj(cross_feat).squeeze(-1)

        q_valid = ~remove_cls_eos_from_mask(q_mask)  # (B, M+2)
        c_valid = ~remove_cls_eos_from_mask(c_mask)  # (B, N+2), but should cls and eos be ignored in reranking?
        cross_mask = torch.einsum('bm,bn->bmn', q_valid, c_valid)  # (B, M+2, N+2)

        local_reranking_logits = self.get_local_logits(cross_feat, q_valid, c_valid, cross_mask)

        q_cls, c_cls = q_emb[:, 0], c_emb[:, 0]
        cls_emb = torch.cat((q_cls, c_cls), dim=1)
        global_reranking_logits = self.global_rerank_proj(cls_emb).squeeze(1)

        reranking_logits = (local_reranking_logits + global_reranking_logits) / 2
        
        return pairing_logits, reranking_logits
    

class RNAflashA(LightningModule):
    def __init__(self, ss_ckpt=None, ss_config=None, dim=256, lr=1e-4, reranking=True, ali=True,
                 margin=0.5, ali_weight=1, use_hard=False, aug_iter=1):
        super().__init__()
        if ss_config is None:
            ss_model = SSModel.load_from_checkpoint(ss_ckpt, map_location='cpu', weights_only=False)
            ss_config = config_to_dict(ss_model.config)
        else:
            cfg = ModelConfig.from_dict(ss_config) if isinstance(ss_config, dict) else ss_config
            ss_model = SSModel(cfg)
            ss_config = config_to_dict(cfg)

        self.save_hyperparameters(ignore=['ss_ckpt'])

        # frozen SSModel backbone + sshead
        for p in ss_model.parameters():
            p.requires_grad = False
        self.backbone = ss_model.backbone
        self.sshead = ss_model.head

        self.head = CrossFeatExtractor(self.backbone.bert_dim, dim)
        self.reranking_bce_fn = BCEWithLogitsLoss()
        self.ali_loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([100]), reduction='none')

        self.lr = lr
        self.margin = margin
        self.reranking = reranking
        self.ali = ali
        self.ali_weight = ali_weight
        self.use_hard = use_hard
        self.aug_iter = aug_iter


    def forward(self, q_toks, q_mask, c_toks, c_mask):
        with torch.no_grad():
            q_emb, q_attn = self.backbone(q_toks, q_mask, get_attn=True)
            c_emb, c_attn = self.backbone(c_toks, c_mask, get_attn=True)
            q_ss_logits = self.sshead(q_attn, q_mask).squeeze(1)
            c_ss_logits = self.sshead(c_attn, c_mask).squeeze(1)
        
        pairing_logits, reranking_logits = self.head.forward(q_emb, q_mask, c_emb, c_mask)  
        
        pairing_logits = self._augment_alignment(pairing_logits, q_ss_logits, c_ss_logits,
                                                    q_mask, c_mask)
        return pairing_logits, reranking_logits

    def _augment_alignment(self, pairing_logits, q_ss_logits, c_ss_logits, q_mask, c_mask):
        q_ss_logits = F.pad(q_ss_logits, (1, 1, 1, 1), mode='constant', value=-1)
        c_ss_logits = F.pad(c_ss_logits, (1, 1, 1, 1), mode='constant', value=-1)
        q_ss_pairing = (q_ss_logits > 0).detach().to(q_ss_logits.dtype) # (bsz, N+2, N+2)
        c_ss_pairing = (c_ss_logits > 0).detach().to(q_ss_logits.dtype)

        for _ in range(self.aug_iter):
            pairing_logits = pairing_logits + q_ss_pairing @ pairing_logits @ c_ss_pairing
        return pairing_logits

    # ------------------------------------------------------------------ losses
    def _triplet_loss(self, pos_logits, neg_logits):
        if self.use_hard:
            hard_neg = neg_logits.max()
            return F.relu(hard_neg - pos_logits + self.margin).mean()
        return F.relu(neg_logits.unsqueeze(1) - pos_logits + self.margin).mean()

    def _get_ali_loss(self, pairing_logits, aln_mask, true_pair):
        ali_loss = self.ali_loss_fn(pairing_logits, true_pair)
        ali_loss = (ali_loss * aln_mask).sum(dim=(1, 2)) / aln_mask.sum(dim=(1, 2))
        ali_loss = ali_loss.mean()
        ali_F1, ali_prec, ali_rec = self._get_ali_F1(pairing_logits, true_pair, aln_mask)
        return ali_loss, ali_F1, ali_prec, ali_rec

    def _get_reranking_loss(self, pos_logits, neg_logits):
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        homo_label = torch.ones_like(pos_logits)
        nonhomo_label = torch.zeros_like(neg_logits)
        labels = torch.cat([homo_label, nonhomo_label], dim=0)
        reranking_bce = self.reranking_bce_fn(logits, labels)
        triple_loss = self._triplet_loss(pos_logits, neg_logits)
        homo_acc = ((pos_logits >= 0) == homo_label.to(bool)).to(float).mean()
        nonhomo_acc = ((neg_logits >= 0) == nonhomo_label.to(bool)).to(float).mean()
        return reranking_bce, triple_loss, homo_acc, nonhomo_acc

    def _get_ali_F1(self, pairing_logits, true_pair, aln_mask):
        bsz = pairing_logits.shape[0]
        F1_scores, prec, rec = [], [], []
        for i in range(bsz):
            m = aln_mask[i].detach().cpu()
            M1 = m[:, 1].sum().item()
            N1 = m[1, :].sum().item()
            logits = pairing_logits[i].detach().cpu()[1:M1 + 1, 1:N1 + 1]
            true_pairs = true_pair[i].detach().cpu()[1:M1 + 1, 1:N1 + 1]
            x, y = torch.where(true_pairs)
            true_set = set(zip(x.tolist(), y.tolist()))
            # single_pred_pairs = decode_optimal_pairs(F.sigmoid(logits))
            single_pred_pairs = decode_optimal_pairs(logits)
            single_pred_pairs = [(r+1, c+1) for r, c in single_pred_pairs]
            pred_set = set(single_pred_pairs)
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            F1_scores.append(f1); prec.append(p); rec.append(r)
        return (sum(F1_scores) / len(F1_scores), sum(prec) / len(prec), sum(rec) / len(rec))

    # ------------------------------------------------------------------ steps
    def training_step(self, batch):
        anchor_toks, pos_toks, neg_toks, anchor_mask, pos_mask, neg_mask, true_aln, aln_mask = batch
        ali_logits, pos_logits = self.forward(anchor_toks, anchor_mask, pos_toks, pos_mask)
        _, neg_logits = self.forward(anchor_toks, anchor_mask, neg_toks, neg_mask)

        if self.ali:
            ali_loss, ali_F1, ali_prec, ali_rec = self._get_ali_loss(ali_logits, aln_mask, true_aln)
            self.log('train/ali_loss', ali_loss, on_epoch=True, on_step=False)
            self.log('train/ali_F1', ali_F1, on_epoch=True, on_step=False)

        if self.reranking:
            reranking_bce, triple_loss, homo_acc, nonhomo_acc = self._get_reranking_loss(pos_logits, neg_logits)
            reranking_acc = (homo_acc + nonhomo_acc) / 2
            self.log('train/reranking_bce', reranking_bce, on_epoch=True, on_step=False)
            self.log('train/reranking_triple_loss', triple_loss, on_epoch=True, on_step=False)
            self.log('train/reranking_homo_acc', homo_acc, on_epoch=True, on_step=False)
            self.log('train/reranking_nonhomo_acc', nonhomo_acc, on_epoch=True, on_step=False)

        if self.reranking and not self.ali:
            loss = reranking_bce + triple_loss
        elif not self.reranking and self.ali:
            loss = ali_loss * self.ali_weight
        elif self.reranking and self.ali:
            loss = reranking_bce + triple_loss + ali_loss * self.ali_weight
        return loss

    def validation_step(self, batch):
        anchor_toks, pos_toks, neg_toks, anchor_mask, pos_mask, neg_mask, true_aln, aln_mask = batch
        ali_logits, pos_logits = self.forward(anchor_toks, anchor_mask, pos_toks, pos_mask)
        _, neg_logits = self.forward(anchor_toks, anchor_mask, neg_toks, neg_mask)
        if self.ali:
            ali_loss, ali_F1, ali_prec, ali_rec = self._get_ali_loss(ali_logits, aln_mask, true_aln)
            self.log('val/ali_loss', ali_loss, on_epoch=True, on_step=False)
            self.log('val/ali_F1', ali_F1, on_epoch=True, on_step=False)
        if self.reranking:
            reranking_bce, triple_loss, homo_acc, nonhomo_acc = self._get_reranking_loss(pos_logits, neg_logits)
            reranking_acc = (homo_acc + nonhomo_acc) / 2
            self.log('val/reranking_bce', reranking_bce, on_epoch=True, on_step=False)
            self.log('val/reranking_triple_loss', triple_loss, on_epoch=True, on_step=False)
            self.log('val/reranking_homo_acc', homo_acc, on_epoch=True, on_step=False)
            self.log('val/reranking_nonhomo_acc', nonhomo_acc, on_epoch=True, on_step=False)
        if self.reranking and not self.ali:
            self.log('val/ali_reranking_ind', reranking_acc, on_epoch=True, on_step=False)
            loss = reranking_bce + triple_loss
        elif not self.reranking and self.ali:
            self.log('val/ali_reranking_ind', ali_F1, on_epoch=True, on_step=False)
            loss = ali_loss * self.ali_weight
        elif self.reranking and self.ali:
            self.log('val/ali_reranking_ind', (reranking_acc + ali_F1) / 2, on_epoch=True, on_step=False)
            loss = reranking_bce + triple_loss + ali_loss * self.ali_weight
        return loss

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        return Adam(trainable, lr=self.lr)

    def on_train_batch_start(self, batch, batch_idx):
        self.backbone.eval()
        self.sshead.eval()


    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def reranking_ali_filter(self, query_seq, targets, converter, threshold=0.5, bsz=16,
                             remove_query_gap=True, ss_aug=True):
        target_names, target_seqs = zip(*targets)
        query_toks, query_mask, _ = converter([('', query_seq)])
        query_toks = query_toks.to(self.device)
        query_mask = query_mask.to(self.device)
        q_emb, q_attn = self.backbone.forward(query_toks, query_mask, get_attn=True)

        reranked_targets = []
        for i in range(0, len(targets), bsz):
            batch = targets[i:i + bsz]
            cur = len(batch)
            tseqs = [t for _, t in batch]
            target_toks, target_mask, _ = converter([('', s) for s in tseqs])
            target_toks = target_toks.to(self.device)
            target_mask = target_mask.to(self.device)
            t_emb, t_attn = self.backbone.forward(target_toks, target_mask, get_attn=True)
            q_emb_b = q_emb.expand(cur, -1, -1)
            q_attn_b = q_attn.expand(cur, -1, -1, -1)
            q_mask_b = query_mask.expand(cur, -1)
            pairing_logits, reranking_logits = self.head.forward(q_emb_b, q_mask_b, t_emb, target_mask)
            if ss_aug:
                q_ss = self.sshead(q_attn_b, q_mask_b).squeeze(1)
                c_ss = self.sshead(t_attn, target_mask).squeeze(1)
                pairing_logits = self._augment_alignment(pairing_logits, q_ss, c_ss, q_mask_b, target_mask)
            homo_probs = F.sigmoid(reranking_logits)
            valid_ids = torch.where(homo_probs >= threshold)[0].cpu().tolist()
            for j in valid_ids:
                target_len = len(tseqs[j])
                single_logits = pairing_logits[j].detach().cpu()[1:len(query_seq) + 1, 1:target_len + 1]
                # pairs = decode_optimal_pairs(F.sigmoid(single_logits))
                pairs = decode_optimal_pairs(single_logits)
                q_gap, t_gap = build_alignment(query_seq, tseqs[j], pairs)
                if remove_query_gap:
                    q_gap, t_gap = apply_remove_query_gap(q_gap, t_gap)
                    reranked_targets.append((batch[j][0], t_gap, homo_probs[j]))
                else:
                    reranked_targets.append((q_gap, batch[j][0], t_gap, homo_probs[j]))
        reranked_targets.sort(key=lambda kv: kv[-1], reverse=True)
        return reranked_targets

    @torch.no_grad()
    def align_batch(self, query_seq, targets, converter, bsz=16, ss_aug=True):
        target_names, target_seqs = zip(*targets)
        query_toks, query_mask, _ = converter([('', query_seq)])
        query_toks = query_toks.to(self.device)
        query_mask = query_mask.to(self.device)
        q_emb, q_attn = self.backbone.forward(query_toks, query_mask, get_attn=True)

        aligned_targets = []
        for i in range(0, len(targets), bsz):
            batch = targets[i:i + bsz]
            cur = len(batch)
            tseqs = [t for _, t in batch]
            target_toks, target_mask, _ = converter([('', s) for s in tseqs])
            target_toks = target_toks.to(self.device)
            target_mask = target_mask.to(self.device)
            t_emb, t_attn = self.backbone.forward(target_toks, target_mask, get_attn=True)
            q_emb_b = q_emb.expand(cur, -1, -1)
            q_attn_b = q_attn.expand(cur, -1, -1, -1)
            q_mask_b = query_mask.expand(cur, -1)
            pairing_logits, _ = self.head.forward(q_emb_b, q_mask_b, t_emb, target_mask)
            if ss_aug:
                q_ss = self.sshead(q_attn_b, q_mask_b).squeeze(1)
                c_ss = self.sshead(t_attn, target_mask).squeeze(1)
                pairing_logits = self._augment_alignment(pairing_logits, q_ss, c_ss, q_mask_b, target_mask)
            for j in range(cur):
                target_len = len(tseqs[j])
                single_logits = pairing_logits[j].detach().cpu()[1:len(query_seq) + 1, 1:target_len + 1]
                # pairs = decode_optimal_pairs(F.sigmoid(single_logits))
                pairs = decode_optimal_pairs(single_logits)
                aligned_targets.append(build_alignment(query_seq, tseqs[j], pairs))
        return aligned_targets
    

    @torch.no_grad()
    def reranking_score(self, q_toks, c_toks, q_mask, c_mask):
        """
        Used for testing only. For application, see reranking_ali_filter.
        The scoring head need no augmentation
        """
        bsz = c_toks.shape[0]
        # assert q_toks.dim() == 2 or q_toks.shape[0] == 1
        assert q_toks.dim() == 2 and q_toks.shape[0] == 1
        q_emb = self.backbone(q_toks, q_mask, use_ckpt=False)
        c_emb = self.backbone(c_toks, c_mask, use_ckpt=False)
        q_emb = q_emb.expand(bsz, -1, -1)
        q_mask = q_mask.expand(bsz, -1)
        _, reranking_logits = self.head.forward(
            q_emb, q_mask, c_emb, c_mask
        )
        homo_probs = F.sigmoid(reranking_logits)
        return homo_probs.cpu().tolist()
    


