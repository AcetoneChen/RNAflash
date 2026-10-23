import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from .rinalmo.pretrained import get_pretrained_model
from .fm.pretrained import from_pretrained
from .ernierna.model import ErnieRNAModel


class Backbone(nn.Module):
    def __init__(self, pretrained_model, bibert=False, freeze=False, *args, **kwargs):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.bibert = bibert
        self.freeze_bert = freeze
        if self.pretrained_model == 'fm':
            self.bert_dim = 640
            self.bert, self.alphabet = from_pretrained("/amax/btchen/RNAhomo/pretrained/FM/RNA-FM_pretrained.pth")
            self.pad_idx = self.bert.padding_idx
        elif self.pretrained_model == 'rinalmo' or self.pretrained_model == 'rinalmo-mega':
            self.bert_dim = 640
            self.bert, self.alphabet = get_pretrained_model('/amax/btchen/RNAhomo/pretrained/rinalmo/rinalmo_mega_pretrained.pt', 'mega')
            self.pad_idx = self.bert.pad_tkn_idx
        elif self.pretrained_model == 'rinalmo-micro':
            self.bert_dim = 480
            self.bert, self.alphabet = get_pretrained_model('/amax/btchen/RNAhomo/pretrained/rinalmo/rinalmo_micro_pretrained.pt', 'micro')
            self.pad_idx = self.bert.pad_tkn_idx
        elif self.pretrained_model == 'rinalmo-giga':
            self.bert_dim = 1280
            self.bert, self.alphabet = get_pretrained_model('/amax/btchen/RNAhomo/pretrained/rinalmo/rinalmo_giga_pretrained.pt', 'giga')
            for i, block in enumerate(self.bert.transformer.blocks):
                if i < 27:
                    for param in block.parameters():
                        param.requires_grad = False
            self.pad_idx = self.bert.pad_tkn_idx
        elif self.pretrained_model == 'ernie_rna':
            self.bert_dim = 768
            self.bert = ErnieRNAModel.from_pretrained('/amax/btchen/RNAhomo/pretrained/ERNIE-RNA/ERNIE-RNA_pretrain.pt')
            self.alphabet = None
        elif self.pretrained_model == 'RNAErnie':
            self.bert_dim = 768
            from transformers import AutoModelForMaskedLM
            self.bert = AutoModelForMaskedLM.from_pretrained('/amax/btchen/RNAhomo/pretrained/RNAErnie').bert
            self.bert.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.alphabet = None
        else:
            raise NotImplementedError

        assert not (self.bibert and self.freeze_bert), "Do not use bi-bert if you don't finetune!"
        if self.bibert:
            raise NotImplementedError
        if self.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, toks, pad_mask, get_attn=False, use_ckpt=True):
        if self.freeze_bert:
            self.bert.eval()
        if self.pretrained_model == 'fm':
            info = self.bert(toks, repr_layers=[12], need_head_weights=get_attn, use_ckpt=use_ckpt)
            emb = info['representations'][12]
        elif 'rinalmo' in self.pretrained_model:
            info = self.bert(toks, need_attn_weights=get_attn, use_checkpoint=use_ckpt)
            emb = info['representation']
        elif self.pretrained_model == 'ernie_rna':
            if get_attn:
                emb, attn_map = self.bert(toks, use_checkpoint=use_ckpt, return_attn=True)
                return emb, attn_map
            emb = self.bert(toks, use_checkpoint=use_ckpt)
        elif self.pretrained_model == 'RNAErnie':
            emb = self.bert(toks, attention_mask=pad_mask).last_hidden_state
        return emb
    
