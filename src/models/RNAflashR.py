import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from lightning import LightningModule
from copy import deepcopy

from .backbone import Backbone
from .taskhead import TaskHead
from ..utils import ModelConfig


class RNAflashR(LightningModule):
    def __init__(self, config, DP=False, use_ss=False):
        super().__init__()
        # Accept a plain dict (as stored in converted checkpoints' hyper_parameters) so
        # RNAflashR.load_from_checkpoint(path) works with the default weights_only=True;
        # a ModelConfig object keeps working unchanged. See convert_legacy_ckpt.py.
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        self.save_hyperparameters()
        self.config = config
        self.DP = DP

        # query encoder (trainable)
        self.use_ss = use_ss
        self.backbone_q = Backbone(config.pretrained_model, bibert=False, freeze=config.freezebert)
        self.taskhead_q = TaskHead(config.taskhead, self.backbone_q.bert_dim, bihead=False)

        # key encoder (momentum updated)
        self.backbone_k = deepcopy(self.backbone_q)
        self.taskhead_k = deepcopy(self.taskhead_q)
        for p in self.backbone_k.parameters():
            p.requires_grad = False
        for p in self.taskhead_k.parameters():
            p.requires_grad = False

        # optimization + moco hyper-parameters
        self.lr = config.lr
        self.moco_m = getattr(config, 'moco_m', 0.999)
        self.moco_T = getattr(config, 'moco_T', 0.07)
        if self.moco_T == 0:
            self.log_moco_T = nn.Parameter(torch.log(torch.tensor(0.07)))
            self.moco_T = torch.exp(self.log_moco_T).item()
        self.moco_K = int(getattr(config, 'moco_K', 65536))
        self.moco_warmup_steps = int(getattr(config, 'moco_warmup_steps', 0))

        head_dim = getattr(config.taskhead, 'd_model', self.backbone_q.bert_dim)
        self.register_buffer('queue', torch.zeros(head_dim, self.moco_K))
        self.register_buffer('queue_family_ids', torch.full((self.moco_K,), -1, dtype=torch.long))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('queue_filled', torch.zeros(1, dtype=torch.long)) # record the number of real data in the queue

        self.q_feats = []
        self.c_feats = []

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for p_q, p_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            p_k.data = p_k.data * self.moco_m + p_q.data * (1.0 - self.moco_m)
        for p_q, p_k in zip(self.taskhead_q.parameters(), self.taskhead_k.parameters()):
            p_k.data = p_k.data * self.moco_m + p_q.data * (1.0 - self.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, fam_ids):
        keys = keys.detach()
        fam_ids = fam_ids.detach().to(dtype=torch.long)
        batch_size = keys.shape[0]
        k = self.queue.shape[1]

        ptr = int(self.queue_ptr.item())
        end = ptr + batch_size

        if end <= k:
            self.queue[:, ptr:end] = keys.T
            self.queue_family_ids[ptr:end] = fam_ids
        else:
            first = k - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, :end - k] = keys[first:].T
            self.queue_family_ids[ptr:] = fam_ids[:first]
            self.queue_family_ids[:end - k] = fam_ids[first:]

        self.queue_ptr[0] = end % k
        self.queue_filled[0] = min(k, int(self.queue_filled.item()) + batch_size)

    def _split_batch(self, batch):
        if isinstance(batch, (tuple, list)) and len(batch) == 5:
            q_toks, c_toks, q_mask, c_mask, fam_ids = batch
            return (q_toks, c_toks, q_mask, c_mask), fam_ids
        if isinstance(batch, (tuple, list)) and len(batch) == 4:
            q_toks, c_toks, q_mask, c_mask = batch
            return (q_toks, c_toks, q_mask, c_mask), None
        raise ValueError('Unexpected batch format for SearchMoCoModel.')

    def _masked_queue_logits(self, q, fam_ids):
        filled = int(self.queue_filled.item())
        if filled <= 0:
            return q.new_empty((q.size(0), 0))

        queue = self.queue[:, :filled].detach()
        l_neg = torch.einsum('nd,dk->nk', q, queue)

        if fam_ids is not None:
            queue_fams = self.queue_family_ids[:filled]
            same_family_mask = fam_ids.view(-1, 1) == queue_fams.view(1, -1)
            l_neg = l_neg.masked_fill(same_family_mask, -1e9)

        return l_neg

    def forward(self, batch, mode):
        q_toks, c_toks, q_mask, c_mask = batch
        assert mode == 'k' or mode == 'q', "mode should be either q or k!"
        if mode == 'k':
            c_emb = self.backbone_k(c_toks, c_mask, False)
            out_c = self.taskhead_k(c_emb, c_mask)
            return out_c
        elif mode == 'q':
            q_emb = self.backbone_q(q_toks, q_mask, False)
            out_q = self.taskhead_q(q_emb, q_mask)
            return out_q
        
    @torch.no_grad()
    def embed(self, toks, mask):
        emb = self.backbone_q(toks, mask, use_ckpt=False)
        out = self.taskhead_q(emb, mask)
        return out

    def training_step(self, batch):
        batch_core, fam_ids = self._split_batch(batch)
        q = self(batch_core, 'q')

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self(batch_core, 'k')

        if fam_ids is None:
            fam_ids = torch.full((q.size(0),), -1, dtype=torch.long, device=q.device)
        else:
            fam_ids = fam_ids.to(device=q.device, dtype=torch.long)

        l_pos = torch.einsum('nd,nd->n', q, k).unsqueeze(-1)
        inbatch_sim = torch.einsum('nd,md->nm', q, k)
        inbatch_mask = torch.eye(inbatch_sim.size(0), device=inbatch_sim.device, dtype=torch.bool)
        inbatch_sim = inbatch_sim.masked_fill(inbatch_mask, -1e9)
        same_family_inbatch = fam_ids.view(-1, 1) == fam_ids.view(1, -1)
        inbatch_sim = inbatch_sim.masked_fill(same_family_inbatch & ~inbatch_mask, -1e9)

        queue_warmed = (int(self.queue_filled.item()) > 0) and (self.global_step >= self.moco_warmup_steps)
        if queue_warmed:
            queue_neg = self._masked_queue_logits(q, fam_ids)
            l_neg = torch.cat([inbatch_sim, queue_neg], dim=1)
        else:
            l_neg = inbatch_sim

        if self.global_step >= self.moco_warmup_steps - 1:
            self._dequeue_and_enqueue(k, fam_ids)

        if hasattr(self, 'log_moco_T'):
            logits = torch.cat([l_pos, l_neg], dim=1) / torch.exp(self.log_moco_T).clip(0.01)
            self.moco_T = torch.exp(self.log_moco_T).item()
            self.log('tau', self.moco_T, on_step=True, on_epoch=True, sync_dist=self.DP)
        else:
            logits = torch.cat([l_pos, l_neg], dim=1) / self.moco_T
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device) # the first is positive, rest negative

        loss = F.cross_entropy(logits, labels)
        top1 = (logits.argmax(dim=1) == labels).float().mean()

        # self._dequeue_and_enqueue(k, fam_ids)

        nonhomo_mask = ~torch.eye(q.size(0), device=self.device, dtype=bool)
        nonhomo_cos = torch.mm(q, k.T)[nonhomo_mask].mean()
        self.log('train/nonhomo_cos', nonhomo_cos, on_step=True, on_epoch=True, sync_dist=self.DP)
        center_loss = 1 - torch.mean(F.cosine_similarity(q, k))
        self.log('train/center_loss', center_loss, on_step=False, on_epoch=True, sync_dist=self.DP)

        self.log('train/loss', loss, on_step=True, on_epoch=True, sync_dist=self.DP)
        self.log('train/top1', top1, on_step=True, on_epoch=True, sync_dist=self.DP)
        self.log('train/queue_filled', float(self.queue_filled.item()), on_step=False, on_epoch=True, sync_dist=self.DP)
        return loss
    
    def on_validation_epoch_start(self):
        self.q_feats = []
        self.c_feats = []


    def validation_step(self, batch):
        batch_core, fam_ids = self._split_batch(batch)
        q = self(batch_core, mode='q')
        k = self(batch_core, mode='k')
        self.q_feats.append(q)
        self.c_feats.append(k)
        loss = self.contrastive_loss_cos(q, k)
        center_loss = 1 - torch.mean(F.cosine_similarity(q, k))
        nonhomo_mask = ~torch.eye(q.size(0), device=self.device, dtype=bool)
        nonhomo_cos = torch.mm(q, k.T)[nonhomo_mask].mean()
        self.log('val/loss', loss, on_step=True, on_epoch=True, sync_dist=self.DP)
        self.log('val/nonhomo_cos', nonhomo_cos, on_step=True, on_epoch=True, sync_dist=self.DP)
        self.log('val/center_loss', center_loss, on_step=False, on_epoch=True, sync_dist=self.DP)

    def configure_optimizers(self):
        if hasattr(self, 'log_moco_T'):
            opt = Adam([
                {'params': self.backbone_q.parameters()},
                {'params': self.taskhead_q.parameters()},
                {'params': [self.log_moco_T], 'lr': 1e-4}
            ], self.lr)
        else:
            opt = Adam(list(self.backbone_q.parameters()) + list(self.taskhead_q.parameters()), lr=self.lr)
        return opt

    def contrastive_loss_cos(self, query_ebds, candidate_ebds):
        if hasattr(self, 'log_moco_T'):
            self.moco_T = torch.exp(self.log_moco_T).item()
        cos_sim = torch.mm(query_ebds, candidate_ebds.T) / self.moco_T
        labels = torch.arange(cos_sim.shape[0], device=cos_sim.device)
        CE_h = F.cross_entropy(cos_sim, labels, reduction='mean')
        CE_v = F.cross_entropy(cos_sim.T, labels, reduction='mean')
        return (CE_h + CE_v) / 2

    def homo_acc(self, query_ebds, candidate_ebds):
        cos_sim = torch.mm(query_ebds, candidate_ebds.T)
        labels = torch.arange(cos_sim.shape[0], device=cos_sim.device)
        predicted = torch.argmax(cos_sim, dim=1)
        acc = (labels == predicted).sum() / cos_sim.size(0)
        return acc
