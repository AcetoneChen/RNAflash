# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .modules import (
    TransformerLayer,
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    RobertaLMHead,
    ESM1bLayerNorm,
    ContactPredictionHead,
)


class BioBertModel(nn.Module):   #ProteinBertModel(nn.Module):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--num_layers", default=36, type=int, metavar="N", help="number of layers"
        )
        parser.add_argument(
            "--embed_dim", default=1280, type=int, metavar="N", help="embedding dimension"
        )
        parser.add_argument(
            "--logit_bias", action="store_true", help="whether to apply bias to logits"
        )
        parser.add_argument(
            "--ffn_embed_dim",
            default=5120,
            type=int,
            metavar="N",
            help="embedding dimension for FFN",
        )
        parser.add_argument(
            "--attention_heads",
            default=20,
            type=int,
            metavar="N",
            help="number of attention heads",
        )

    def __init__(self, args, alphabet):
        super().__init__()
        self.args = args
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.emb_layer_norm_before = getattr(self.args, "emb_layer_norm_before", False)
        if self.args.arch == "roberta_large":
            self.model_version = "ESM-1b"
            self._init_submodules_esm1b()
        else:
            self.model_version = "ESM-1"
            self._init_submodules_esm1()

    def _init_submodules_common(self):
        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.args.embed_dim, padding_idx=self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.args.embed_dim,
                    self.args.ffn_embed_dim,
                    self.args.attention_heads,
                    add_bias_kv=(self.model_version != "ESM-1b"),
                    use_esm1b_layer_norm=(self.model_version == "ESM-1b"),
                )
                for _ in range(self.args.layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.args.layers * self.args.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )

    def _init_submodules_esm1b(self):
        self._init_submodules_common()
        self.embed_scale = 1
        self.embed_positions = LearnedPositionalEmbedding(
            self.args.max_positions, self.args.embed_dim, self.padding_idx
        )
        self.emb_layer_norm_before = (
            ESM1bLayerNorm(self.args.embed_dim) if self.emb_layer_norm_before else None
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=self.args.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def _init_submodules_esm1(self):
        self._init_submodules_common()
        self.embed_scale = math.sqrt(self.args.embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(self.args.embed_dim, self.padding_idx)
        self.embed_out = nn.Parameter(torch.zeros((self.alphabet_size, self.args.embed_dim)))
        self.embed_out_bias = None
        if self.args.final_bias:
            self.embed_out_bias = nn.Parameter(torch.zeros(self.alphabet_size))

    def forward(
            self, 
            tokens, 
            repr_layers=[], 
            need_head_weights=False, 
            return_contacts=False, 
            use_ckpt=True,
            attn_bias=None
        ):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if getattr(self.args, "token_dropout", False):
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).float() / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        x = x + self.embed_positions(tokens)

        if self.model_version == "ESM-1b":
            if self.emb_layer_norm_before:
                x = self.emb_layer_norm_before(x)
            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            def layer_forward(x, padding_mask, need_head_weights):
                return layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_head_weights=need_head_weights,
                    self_attn_mask=attn_bias
                )
            if use_ckpt:
                if layer_idx != len(self.layers) - 1:
                    x, attn = checkpoint(layer_forward, x, padding_mask, False, use_reentrant=False)
                else:
                    x, attn = checkpoint(layer_forward, x, padding_mask, need_head_weights, use_reentrant=False)
            else:
                x, attn = layer_forward(x, padding_mask, need_head_weights)
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)

        if self.model_version == "ESM-1b":
            x = self.emb_layer_norm_after(x)
            x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

            # last hidden representation should have layer norm applied
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x
            x = self.lm_head(x, None)    # CJY at add masked_tokens for lm_head
        else:
            x = F.linear(x, self.embed_out, bias=self.embed_out_bias)
            x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        result = {"logits": x, "representations": hidden_representations}
        # if need_head_weights:
        result['attentions'] = attn

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    @property
    def num_layers(self):
        return self.args.layers
