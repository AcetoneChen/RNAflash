"""
Self-contained ERNIE-RNA model for RNA sequence representation extraction.
No external dependencies beyond torch and numpy.

Usage:
    from src2.models.ernierna import ErnieRNAModel

    model = ErnieRNAModel.from_pretrained(
        'checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt'
    )
    model.eval()

    # tokens: [B, L] long tensor with CLS=0, PAD=1, EOS=2
    repr = model(tokens)             # [B, L, 768]
    repr = model(tokens, pad_mask)   # with explicit padding mask
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
from torch.utils.checkpoint import checkpoint



# ==============================================================================
# 2D matrix helpers (base-pairing propensity matrix)
# ==============================================================================

def _gaussian(x: float) -> float:
    return math.exp(-0.5 * (x * x))


def _paired(x: int, y: int, lamda: float = 0.8) -> float:
    """Return base-pairing score for two nucleotide indices."""
    if x == 5 and y == 6:    # A-U
        return 2
    elif x == 4 and y == 7:  # G-C
        return 3
    elif x == 4 and y == 6:  # G-U
        return lamda
    elif x == 6 and y == 5:  # U-A
        return 2
    elif x == 7 and y == 4:  # C-G
        return 3
    elif x == 6 and y == 4:  # U-G
        return lamda
    else:
        return 0


def creatmat(data: np.ndarray, base_range: int = 1, lamda: float = 0.8) -> np.ndarray:
    """Build 2D structural pairing matrix for an RNA sequence.

    Args:
        data: [L] integer token indices.
        base_range: pair distance range (1 = direct, 30 = long-range).
        lamda: score for G-U wobble pairs.

    Returns:
        [L, L] float matrix of base-pairing propensities.
    """
    paird_map = np.array([[ _paired(i, j, lamda) for i in range(30) ] for j in range(30)])
    data_index = np.arange(0, len(data))
    coefficient = np.zeros([len(data), len(data)])
    score_mask = np.full((len(data), len(data)), True)

    for add in range(base_range):
        data_index_x = data_index - add
        data_index_y = data_index + add
        score_mask = ((data_index_x >= 0)[:, None] & (data_index_y < len(data))[None, :]) & score_mask
        data_index_x, data_index_y = np.meshgrid(
            data_index_x.clip(0, len(data) - 1),
            data_index_y.clip(0, len(data) - 1),
            indexing='ij',
        )
        score = paird_map[data[data_index_x], data[data_index_y]]
        score_mask = score_mask & (score != 0)
        coefficient = coefficient + score * score_mask * _gaussian(add)
        if ~(score_mask.any()):
            break

    score_mask = coefficient > 0
    for add in range(1, base_range):
        data_index_x = data_index + add
        data_index_y = data_index - add
        score_mask = ((data_index_x < len(data))[:, None] & (data_index_y >= 0)[None, :]) & score_mask
        data_index_x, data_index_y = np.meshgrid(
            data_index_x.clip(0, len(data) - 1),
            data_index_y.clip(0, len(data) - 1),
            indexing='ij',
        )
        score = paird_map[data[data_index_x], data[data_index_y]]
        score_mask = score_mask & (score != 0)
        coefficient = coefficient + score * score_mask * _gaussian(add)
        if ~(score_mask.any()):
            break

    return coefficient


# ==============================================================================
# 2D multi-head attention (standalone, adapted from fairseq's MHA)
# ==============================================================================

def _scaled_dot_product_attention_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    twod_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    if twod_tokens is not None:
        attn += twod_tokens
    twod_tokens_new = attn
    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    output = torch.bmm(attn, v)
    return output, attn, twod_tokens_new


def _multi_head_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    embed_dim: int,
    num_heads: int,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    in_proj_bias: Optional[torch.Tensor],
    out_proj_weight: torch.Tensor,
    out_proj_bias: Optional[torch.Tensor],
    dropout_p: float,
    training: bool,
    key_padding_mask: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    twod_tokens: Optional[torch.Tensor] = None,
    is_twod: bool = False,
    need_weights: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    tgt_len, bsz, embed_dim_q = query.shape
    src_len, _, _ = key.shape
    head_dim = embed_dim // num_heads

    # in-projection (separate Q/K/V)
    if in_proj_bias is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = in_proj_bias.chunk(3)
    q = F.linear(query, q_proj_weight, b_q)
    k = F.linear(key, k_proj_weight, b_k)
    v = F.linear(value, v_proj_weight, b_v)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, num_heads, -1, -1)
        key_padding_mask = key_padding_mask.reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    if is_twod and twod_tokens is not None:
        twod_tokens = twod_tokens.reshape(bsz * num_heads, src_len, src_len)

    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    if not training:
        dropout_p = 0.0

    # reshape: (T, B, C) -> (B*heads, T, head_dim)
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    twod_tokens_new = None
    if is_twod:
        attn_output, attn_output_weights, twod_tokens_new = _scaled_dot_product_attention_2d(
            q, k, v, attn_mask, dropout_p, twod_tokens,
        )
    else:
        attn_output, attn_output_weights = F._scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p,
        )

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if is_twod and twod_tokens_new is not None:
        twod_tokens_new = twod_tokens_new.view(bsz, num_heads, tgt_len, src_len)

    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads, twod_tokens_new
    else:
        return attn_output, None, twod_tokens_new


# ==============================================================================
# Sinusoidal positional embedding
# ==============================================================================

def get_sinusoidal_embedding(
    num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
) -> torch.Tensor:
    """Generate sinusoidal positional embeddings (non-learned)."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(num_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(num_positions, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx] = 0
    return emb


# ==============================================================================
# Model components
# ==============================================================================

class MultiheadAttention(nn.Module):
    """Multi-head attention with separate Q/K/V projections and 2D bias support."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout_p = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        twod_tokens: Optional[torch.Tensor] = None,
        is_twod: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert key is not None and value is not None
        return _multi_head_attention_forward(
            query, key, value,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias])
            if self.q_proj.bias is not None else None,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            dropout_p=self.dropout_p,
            training=self.training,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            twod_tokens=twod_tokens,
            is_twod=is_twod,
            need_weights=need_weights,
        )


class TransformerSentenceEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer."""

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "gelu",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.self_attn = MultiheadAttention(
            embedding_dim, num_attention_heads, dropout=attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(activation_dropout)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

        if activation_fn == "gelu":
            self.activation_fn = F.gelu
        elif activation_fn == "relu":
            self.activation_fn = F.relu
        elif activation_fn == "tanh":
            self.activation_fn = torch.tanh
        else:
            self.activation_fn = F.gelu

    def forward(
        self,
        x: torch.Tensor,
        twod_tokens: torch.Tensor,
        is_twod: bool,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        residual = x
        x, _attn, twod_tokens_new = self.self_attn(
            query=x, key=x, value=x,
            twod_tokens=twod_tokens,
            is_twod=is_twod,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
        )
        x = self.dropout(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        x = self.final_layer_norm(x)

        return x, None, twod_tokens_new


class TransformerSentenceEncoder(nn.Module):
    """BERT-style Transformer encoder with positional embeddings."""

    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_seq_len: int = 1024,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        encoder_normalize_before: bool = True,
        learned_pos_embedding: bool = False,
        activation_fn: str = "gelu",
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        self.embed_tokens = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.dropout = nn.Dropout(dropout)

        self.segment_embeddings = (
            nn.Embedding(num_segments, embedding_dim)
            if num_segments > 0 else None
        )

        self.use_position_embeddings = use_position_embeddings
        self.learned_pos_embedding = learned_pos_embedding
        if use_position_embeddings:
            if learned_pos_embedding:
                # Fairseq PositionalEmbedding(learned=True) adds padding_idx+1 rows
                n_pos = max_seq_len + padding_idx + 1
                self.embed_positions = nn.Embedding(n_pos, embedding_dim, padding_idx)
            else:
                # Sinusoidal: init_size = max_seq_len + padding_idx + 1 (matches fairseq)
                n_pos = max_seq_len + padding_idx + 1
                sin_emb = get_sinusoidal_embedding(n_pos, embedding_dim, padding_idx)
                self.register_buffer("_sinusoidal_pos", sin_emb)
        else:
            self.register_buffer("_sinusoidal_pos", None)

        self.emb_layer_norm = nn.LayerNorm(embedding_dim) if encoder_normalize_before else None

        self.layers = nn.ModuleList([
            TransformerSentenceEncoderLayer(
                embedding_dim=embedding_dim,
                ffn_embedding_dim=ffn_embedding_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
            )
            for _ in range(num_encoder_layers)
        ])

    def _get_position_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.use_position_embeddings:
            return 0
        # Make position indices with padding handling (matches fairseq's make_positions)
        mask = tokens.ne(self.padding_idx).long()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        if self.learned_pos_embedding:
            return self.embed_positions(positions)
        else:
            return F.embedding(positions, self._sinusoidal_pos)

    def _layer_forward(self, layer, x, twod_tokens, is_twod, padding_mask):
        return layer(
            x,
            twod_tokens=twod_tokens,
            is_twod=is_twod,
            self_attn_padding_mask=padding_mask,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: Optional[torch.Tensor] = None,
        is_twod: bool = False,
        twod_tokens: Optional[torch.Tensor] = None,
        last_state_only: bool = False,
        use_ckpt: bool = True,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        # padding mask
        padding_mask = tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        # token embeddings
        x = self.embed_tokens(tokens)

        # positional embeddings
        pos_emb = self._get_position_embedding(tokens)
        if isinstance(pos_emb, torch.Tensor):
            x = x + pos_emb

        # segment embeddings
        if self.segment_embeddings is not None and segment_labels is not None:
            x = x + self.segment_embeddings(segment_labels)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        x = self.dropout(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states: List[torch.Tensor] = []
        if not last_state_only:
            inner_states.append(x)

        twod_tokens_lst: List[Optional[torch.Tensor]] = [twod_tokens]
        for layer in self.layers:
            if use_ckpt:
                x, _, twod_tokens = checkpoint(
                    self._layer_forward,
                    layer, x, twod_tokens, is_twod, padding_mask,
                    use_reentrant=False,
                )
            else:
                x, _, twod_tokens = layer(
                    x,
                    twod_tokens=twod_tokens,
                    is_twod=is_twod,
                    self_attn_padding_mask=padding_mask,
                )
            twod_tokens_lst.append(twod_tokens)
            if not last_state_only:
                inner_states.append(x)

        sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        return inner_states, sentence_rep, twod_tokens_lst


class NonLinearHead(nn.Module):
    """Small 2-layer MLP for projecting 2D input to attention heads."""

    def __init__(self, input_dim: int, out_dim: int, activation_fn: str = "gelu", hidden: int = None):
        super().__init__()
        hidden = input_dim if hidden is None else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        if activation_fn == "gelu":
            self.activation_fn = F.gelu
        elif activation_fn == "relu":
            self.activation_fn = F.relu
        else:
            self.activation_fn = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


# ==============================================================================
# Top-level model
# ==============================================================================

class ErnieRNAModel(nn.Module):
    """
    Self-contained ERNIE-RNA model for inference.

    Loads a fairseq pretrained checkpoint and provides a clean forward()
    for extracting RNA sequence representations.

    Parameters (auto-detected from checkpoint):
        vocab_size: 25
        embed_dim: 768
        num_layers: 12
        num_heads: 12
        ffn_dim: 3072
        max_seq_len: 1024
        dropout: 0.1
        attention_dropout: 0.1
        activation_dropout: 0.0
        num_segments: 2
        padding_idx: 1
    """

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        map_location: str = "cpu",
        base_range: int = 1,
        lamda: float = 0.8,
    ) -> "ErnieRNAModel":
        """Load a pretrained ERNIE-RNA model from a fairseq checkpoint.

        Args:
            checkpoint_path: path to ERNIE-RNA_pretrain.pt.
            map_location: device to load to ("cpu" or "cuda").
            base_range: pairing distance for 2D matrix (1 = direct).
            lamda: G-U wobble pair score.

        Returns:
            ErnieRNAModel instance with loaded weights.
        """
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        state_dict = ckpt["model"]

        # Detect architecture from state_dict shapes
        embed_tokens_w = state_dict["encoder.sentence_encoder.embed_tokens.weight"]
        cls_vocab_size, embed_dim = embed_tokens_w.shape

        # Count layers
        num_layers = 0
        while f"encoder.sentence_encoder.layers.{num_layers}.fc1.weight" in state_dict:
            num_layers += 1

        fc1_w = state_dict[f"encoder.sentence_encoder.layers.0.fc1.weight"]
        ffn_dim = fc1_w.shape[0]

        twod_w = state_dict["encoder.twod_proj.linear2.weight"]
        num_heads = twod_w.shape[0]

        seg_w = state_dict.get("encoder.sentence_encoder.segment_embeddings.weight")
        num_segments = seg_w.shape[0] if seg_w is not None else 0

        max_seq_len = 1024
        padding_idx = 1

        model = cls(
            vocab_size=cls_vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            max_seq_len=max_seq_len,
            num_segments=num_segments,
            padding_idx=padding_idx,
            base_range=base_range,
            lamda=lamda,
        )

        # Load state dict (strict=False to skip sinusoidal sentinel _float_tensor)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[ErnieRNAModel] Missing keys (expected, non-learned): {missing}")
        if unexpected:
            print(f"[ErnieRNAModel] Unexpected keys: {unexpected}")

        model.to(map_location)
        return model

    def __init__(
        self,
        vocab_size: int = 25,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        num_segments: int = 2,
        padding_idx: int = 1,
        activation_fn: str = "gelu",
        pooler_activation_fn: str = "tanh",
        base_range: int = 1,
        lamda: float = 0.8,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.base_range = base_range
        self.lamda = lamda

        # Build encoder container (mirrors fairseq key structure)
        encoder = nn.Module()
        encoder.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=padding_idx,
            vocab_size=vocab_size,
            num_encoder_layers=num_layers,
            embedding_dim=embed_dim,
            ffn_embedding_dim=ffn_dim,
            num_attention_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            max_seq_len=max_seq_len,
            num_segments=num_segments,
            use_position_embeddings=True,
            encoder_normalize_before=True,
            learned_pos_embedding=False,
            activation_fn=activation_fn,
        )

        # MLM pooler (unused during inference, but needed for state_dict loading)
        encoder.masked_lm_pooler = nn.Linear(embed_dim, embed_dim)
        encoder.lm_head_transform_weight = nn.Linear(embed_dim, embed_dim)
        encoder.layer_norm = nn.LayerNorm(embed_dim)
        encoder.lm_output_learned_bias = nn.Parameter(torch.zeros(vocab_size))

        # 2D projection: [B, L, L, 1] -> [B, L, L, num_heads]
        encoder.twod_proj = NonLinearHead(
            input_dim=1, out_dim=num_heads, activation_fn=activation_fn, hidden=6,
        )

        self.encoder = encoder

        # Per-(device, lamda) cache of the 30x30 base-pair score table.
        self._paird_cache: dict = {}

        # Activation functions (stored for reference, not registered as modules)
        if pooler_activation_fn == "tanh":
            self._pooler_activation = torch.tanh
        elif pooler_activation_fn == "gelu":
            self._pooler_activation = F.gelu
        else:
            self._pooler_activation = torch.tanh

    def _get_paird_map(self, device: torch.device) -> torch.Tensor:
        """Cached 30x30 base-pair score table on the requested device.

        Layout matches the ``paird_map`` built inside :func:`creatmat`; for
        ``base_range == 1`` a single gather reproduces the full matrix exactly.
        """
        key = (str(device), self.lamda)
        pm = self._paird_cache.get(key)
        if pm is None:
            table = np.array(
                [[_paired(i, j, self.lamda) for i in range(30)] for j in range(30)],
                dtype=np.float32,
            )
            pm = torch.from_numpy(table).to(device)
            self._paird_cache[key] = pm
        return pm

    def _compute_2d_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute 2D structural matrix from token indices.

        Args:
            tokens: [B, L] long tensor.

        Returns:
            [B, L, L, 1] float tensor.
        """
        B, L = tokens.shape

        if self.base_range == 1:
            # Fast path: the matrix is exactly the pairwise base-pair score
            # table lookup, so one batched gather on the tokens' own device
            # replaces the per-sample Python/numpy loop (and its CPU sync).
            # Numerically identical to creatmat(tokens, 1, lamda).
            pm = self._get_paird_map(tokens.device)
            twod = pm[tokens[:, :, None], tokens[:, None, :]]  # [B, L, L]
            return twod.unsqueeze(-1)

        # General path (base_range > 1): keep the original per-sample loop.
        tokens_np = tokens.cpu().numpy()
        twod_batch = np.zeros((B, L, L), dtype=np.float32)
        for b in range(B):
            twod_batch[b] = creatmat(tokens_np[b].astype(int),
                                     base_range=self.base_range,
                                     lamda=self.lamda)
        twod = torch.from_numpy(twod_batch).unsqueeze(-1)  # [B, L, L, 1]
        return twod.to(tokens.device)

    def forward(
        self,
        tokens: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
        return_attn: bool = False,
        use_checkpoint: bool = True,
    ) -> torch.Tensor:
        """
        Extract RNA sequence representations.

        Args:
            tokens: [B, L] long tensor. Tokenized RNA with CLS(0), PAD(1), EOS(2).
            pad_mask: [B, L] bool tensor, True = padding position. Auto-computed if None.
            return_all_layers: if True, return [num_layers, B, L, C]; else [B, L, C].
            return_attn: if True, also return the last encoder layer's accumulated
                         attention map [B, num_heads, L, L] (same semantics as the
                         official ERNIE-RNA attn-map SS prediction head input).
            use_ckpt: use activation checkpointing on each transformer layer
                      (saves memory during training, no-op during eval).

        Returns:
            Sequence representation tensor (or (repr, attn_map) when return_attn=True).
        """
        B, L = tokens.shape

        # Compute 2D structural matrix
        twod_input = self._compute_2d_tokens(tokens).float()

        # Project 2D: [B, L, L, 1] -> [B, L, L, num_heads] -> [B, num_heads, L, L]
        twod_proj = self.encoder.twod_proj.float()(twod_input)
        twod_bias = twod_proj.permute(0, 3, 1, 2).contiguous()

        # Pass through sentence encoder
        inner_states, _sentence_rep, _twod_lst = self.encoder.sentence_encoder(
            tokens,
            is_twod=True,
            twod_tokens=twod_bias,
            use_ckpt=use_checkpoint,
        )

        # inner_states: list of 13 tensors [T, B, C]
        #   [0]: embedding output, [1..12]: after each layer
        if return_all_layers:
            # Return all layer outputs except the embedding (idx 0)
            all_layer_states = torch.stack(inner_states[1:])  # [12, T, B, C]
            all_layer_states = all_layer_states.transpose(1, 2)  # [12, B, T, C]
            out = all_layer_states
        else:
            # Default path only needs the last layer; skip the 13-layer stack.
            out = inner_states[-1].transpose(0, 1)  # [B, T, C]

        if return_attn:
            return out, _twod_lst[-1]  # ([B, T, C], [B, num_heads, L, L])
        return out
