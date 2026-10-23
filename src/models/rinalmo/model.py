import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
import math
from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_qkvpacked_func
from flash_attn.layers.rotary import RotaryEmbedding

from flash_attn.bert_padding import unpad_input, pad_input

from einops import rearrange


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        base: int = 10000
    ):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _update_cached(self, x, seq_dim):
        seq_len = x.shape[seq_dim]

        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, q, k):
        self._update_cached(k, seq_dim=-2)
        return apply_rotary_pos_emb(q, k, self.cos_cached, self.sin_cached)
    
def dot_product_attention(q, k, v, attn_mask=None, key_pad_mask=None, dropout=None):
    c = q.shape[-1]
    attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(c)

    if attn_mask is not None:
        attn = attn.masked_fill(attn_mask, float("-inf"))

    if key_pad_mask is not None:
        attn = attn.masked_fill(key_pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

    attn = attn.softmax(dim=-1)
    if dropout is not None:
        attn = dropout(attn)

    output = torch.matmul(attn, v)
    return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, c_in, num_heads, attention_dropout=0.0, use_rot_emb=True, bias=False):
        super().__init__()
        assert c_in % num_heads == 0, "Embedding dimensionality must be divisible with number of attention heads!"

        self.c_in = c_in
        self.num_heads = num_heads

        self.c_head = c_in // self.num_heads
        self.c_qkv = self.c_head * num_heads

        self.use_rot_emb = use_rot_emb
        if self.use_rot_emb:
            self.rotary_emb = RotaryPositionEmbedding(self.c_head)

        self.to_q = nn.Linear(self.c_in, self.c_qkv, bias=bias)
        self.to_k = nn.Linear(self.c_in, self.c_qkv, bias=bias)
        self.to_v = nn.Linear(self.c_in, self.c_qkv, bias=bias)

        self.attention_dropout = nn.Dropout(p=attention_dropout)

        self.out_proj = nn.Linear(c_in, c_in, bias=bias)

    def forward(self, q, k, v, attn_mask=None, key_pad_mask=None):
        bs = q.shape[0]

        q = self.to_q(q).view(bs, -1, self.num_heads, self.c_head).transpose(-2, -3)
        k = self.to_k(k).view(bs, -1, self.num_heads, self.c_head).transpose(-2, -3)
        v = self.to_v(v).view(bs, -1, self.num_heads, self.c_head).transpose(-2, -3)

        if self.use_rot_emb:
            q, k = self.rotary_emb(q, k)

        output, attn = dot_product_attention(q, k, v, attn_mask, key_pad_mask, self.attention_dropout)

        output = output.transpose(-2, -3).contiguous().view(bs, -1, self.num_heads * self.c_head)
        output = self.out_proj(output)

        return output, attn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, c_in, num_heads, attention_dropout=0.0, use_rot_emb=True, bias=False):
        super().__init__()

        self.mh_attn = MultiHeadAttention(c_in, num_heads, attention_dropout, use_rot_emb, bias)

    def forward(self, x, attn_mask=None, key_pad_mask=None):
        return self.mh_attn(x, x, x, attn_mask, key_pad_mask)
    

class FlashAttention(nn.Module):
    """
    Implement the scaled dot product attention with softmax.
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        """
        Args:
            causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
            softmax_scale: float. The scaling of QK^T before applying softmax.
                           Default to 1 / sqrt(headdim).
            attention_dropout: float. The dropout rate to apply to the attention
                               (default: 0.0)
        """
        super().__init__()
        self.softmax_scale = softmax_scale
        self.attention_dropout = attention_dropout
        self.causal = causal

    def forward(self, qkv, cu_seqlens=None, max_seqlen=None, return_attn_probs=False):
        """
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value.
                 If cu_seqlens is None and max_seqlen is None, then qkv has shape (B, S, 3, H, D).
                 If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
                 (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                        of the sequences in the batch, used to index into qkv.
            max_seqlen: int. Maximum sequence length in the batch.
            return_attn_probs: bool. Whether to return the attention probabilities. This option is for
                               testing only. The returned probabilities are not guaranteed to be correct
                               (they might not have the right scaling).
        Returns:
        --------
            out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
                 else (B, S, H, D).
        """
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        unpadded = cu_seqlens is not None

        if unpadded:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            return flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_seqlen,
                self.attention_dropout if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                return_attn_probs=return_attn_probs
            )
        else:
            return flash_attn_qkvpacked_func(
                qkv,
                self.attention_dropout if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                return_attn_probs=return_attn_probs
            )

class FlashMultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention implemented using FlashAttention.
    """
    def __init__(self, embed_dim, num_heads, attention_dropout=0.0, causal=False, use_rot_emb=True, bias=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimensionality must be divisible with number of attention heads!"

        self.causal = causal

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = self.embed_dim // self.num_heads
        self.qkv_dim = self.head_dim * num_heads * 3

        self.rotary_emb_dim = self.head_dim
        self.use_rot_emb = use_rot_emb
        if self.use_rot_emb:
            self.rotary_emb = RotaryEmbedding(
                dim=self.rotary_emb_dim,
                base=10000.0,
                interleaved=False,
                scale_base=None,
                pos_idx_in_fp32=True,  # fp32 RoPE precision
                device=None
            )
        self.flash_self_attn = FlashAttention(causal=self.causal, softmax_scale=None, attention_dropout=attention_dropout)

        self.Wqkv = nn.Linear(self.embed_dim, self.qkv_dim, bias=bias)

        self.attention_dropout = nn.Dropout(p=attention_dropout)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

    def forward(self, x, key_padding_mask=None, return_attn_probs=False):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num_heads * head_dim)
            key_pad_mask: boolean mask, True means to keep, False means to mask out.
                          (batch, seqlen)
            return_attn_probs: whether to return attention masks (False by default)
        """

        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)

        if self.use_rot_emb:
            qkv = self.rotary_emb(qkv, seqlen_offset=0)

        if return_attn_probs:
            bs = qkv.shape[0]
            qkv = torch.permute(qkv, (0, 3, 2, 1, 4))
            q = qkv[:, :, 0, :, :]
            k = qkv[:, :, 1, :, :]
            v = qkv[:, :, 2, :, :]
            out, attn = dot_product_attention(q, k, v, key_pad_mask=torch.logical_not(key_padding_mask) if key_padding_mask is not None else None, dropout=self.attention_dropout)
            output = out.transpose(-2, -3).contiguous().view(bs, -1, self.num_heads * self.head_dim)
            output = self.out_proj(output)
            return output, attn

        if key_padding_mask is not None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            x_unpad, indices, cu_seqlens, max_s, _ = unpad_input(qkv, key_padding_mask)
            output_unpad = self.flash_self_attn(
                    x_unpad, 
                    cu_seqlens=cu_seqlens, 
                    max_seqlen=max_s, 
                    return_attn_probs=return_attn_probs
                    )
            out = pad_input(rearrange(output_unpad, '... h d -> ... (h d)'), indices, batch_size, seqlen)
        else:
            output = self.flash_self_attn(
                    qkv, 
                    cu_seqlens=None, 
                    max_seqlen=None, 
                    return_attn_probs=return_attn_probs
                    )
            out = rearrange(output, '... h d -> ... (h d)')

        out = self.out_proj(out)
        return out, None


class TokenDropout(nn.Module):
    def __init__(
        self,
        active: bool,
        mask_ratio: float,
        mask_tkn_prob: float,
        mask_tkn_idx: int,
        pad_tkn_idx: int,
    ):
        super().__init__()

        self.active = active

        self.mask_ratio_train = mask_ratio * mask_tkn_prob

        self.mask_tkn_idx = mask_tkn_idx
        self.pad_tkn_idx = pad_tkn_idx

    def forward(self, x, tokens):
        if self.active:
            pad_mask = tokens.eq(self.pad_tkn_idx)
            src_lens = (~pad_mask).sum(dim=-1)

            x = torch.where((tokens == self.mask_tkn_idx).unsqueeze(dim=-1), 0.0, x)
            mask_ratio_observed = (tokens == self.mask_tkn_idx).sum(dim=-1) / src_lens
            x = x * (1 - self.mask_ratio_train) / (1 - mask_ratio_observed[..., None, None])

        return x

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_blocks, num_heads, use_rot_emb=True, attn_qkv_bias=False, transition_dropout=0.0, attention_dropout=0.0, residual_dropout=0.0, transition_factor=4, use_flash_attn=False):
        super().__init__()

        self.use_flash_attn = use_flash_attn

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, use_rot_emb, attn_qkv_bias, transition_dropout, attention_dropout, residual_dropout, transition_factor, use_flash_attn) for _ in range(num_blocks)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None, need_attn_weights=False, use_checkpoint=False):
        attn_weights = None
        need_attn_weights = [False] * (len(self.blocks) - 1) + [need_attn_weights]

        for i, block in enumerate(self.blocks):
            if use_checkpoint:
                x, attn = checkpoint.checkpoint(
                    block, 
                    x,
                    key_padding_mask=key_padding_mask,
                    need_attn_weights=need_attn_weights[i],
                    use_reentrant=False
                    )
            else:
                x, attn = block(x, key_padding_mask, need_attn_weights[i]) # if use contrastive, then ckpt not available
            if i == len(self.blocks) - 1:
                attn_weights = attn

        x = self.final_layer_norm(x)

        return x, attn_weights

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit
    https://arxiv.org/pdf/2002.05202v1.pdf
    In the cited paper beta is set to 1 and is not learnable;
    but by the Swish definition it is learnable parameter otherwise
    it is SiLU activation function (https://paperswithcode.com/method/swish)
    """
    def __init__(self, size_in, size_out, beta_is_learnable=True, bias=True):
        """
        Args:
            size_in: input embedding dimension
            size_out: output embedding dimension
            beta_is_learnable: whether beta is learnable or set to 1, learnable by default
            bias: whether use bias term, enabled by default
        """
        super().__init__()
        self.linear = nn.Linear(size_in, size_out, bias=bias)
        self.linear_gate = nn.Linear(size_in, size_out, bias=bias)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=beta_is_learnable)  

    def forward(self, x):
        linear_out = self.linear(x)
        swish_out = linear_out * torch.sigmoid(self.beta * linear_out)
        return swish_out * self.linear_gate(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, use_rot_emb=True, attn_qkv_bias=False, transition_dropout=0.0, attention_dropout=0.0, residual_dropout=0.0, transition_factor=4, use_flash_attn=False):
        super().__init__()
        
        self.use_flash_attn = use_flash_attn

        if use_flash_attn:
            self.mh_attn = FlashMultiHeadSelfAttention(embed_dim, num_heads, attention_dropout, causal=False, use_rot_emb=use_rot_emb, bias=attn_qkv_bias)
        else:
            self.mh_attn = MultiHeadSelfAttention(embed_dim, num_heads, attention_dropout, use_rot_emb, attn_qkv_bias)
        
        self.attn_layer_norm = nn.LayerNorm(embed_dim)

        self.transition = nn.Sequential(
                SwiGLU(embed_dim, int(2 / 3 * transition_factor * embed_dim), beta_is_learnable=True, bias=True),
                nn.Dropout(p=transition_dropout),
                nn.Linear(int(2 / 3 * transition_factor * embed_dim), embed_dim, bias=True),
        )
        self.out_layer_norm = nn.LayerNorm(embed_dim)

        self.residual_dropout_1 = nn.Dropout(p=residual_dropout)
        self.residual_dropout_2 = nn.Dropout(p=residual_dropout)

    def forward(self, x, key_padding_mask=None, need_attn_weights=None):
        x = self.attn_layer_norm(x)
        if self.use_flash_attn:
            mh_out, attn = self.mh_attn(x, key_padding_mask=key_padding_mask, return_attn_probs=need_attn_weights)
        else:
            mh_out, attn = self.mh_attn(x, attn_mask=None, key_pad_mask=key_padding_mask)
        x = x + self.residual_dropout_1(mh_out)

        residual = x
        x = self.out_layer_norm(x)
        x = residual + self.residual_dropout_2(self.transition(x))

        return x, attn

class MaskedLanguageModelHead(nn.Module):
    def __init__(self, embed_dim, alphabet_size):
        super().__init__()

        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear2 = nn.Linear(embed_dim, alphabet_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.linear2(x)

        return x


class RiNALMo(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.pad_tkn_idx = self.config.model.embedding['padding_idx']

        self.embedding = nn.Embedding(**self.config.model['embedding'])
        self.transformer = Transformer(**self.config.model['transformer'])

        self.lm_mask_head = MaskedLanguageModelHead(**self.config.model['lm_mask_head'])

        self.token_dropout = TokenDropout(**self.config.model['token_dropout'])

    def forward(self, tokens, need_attn_weights=False, use_checkpoint=True):
        pad_mask = tokens.eq(self.pad_tkn_idx)
        x = self.embedding(tokens)
        x = self.token_dropout(x, tokens)

        if self.config.model.transformer.use_flash_attn:
            representation, attn_weights = self.transformer(
                x,
                key_padding_mask=torch.logical_not(pad_mask) if pad_mask is not None else None,
                need_attn_weights=need_attn_weights,
                use_checkpoint=use_checkpoint
                )
        else:
            representation, attn_weights = self.transformer(x, key_padding_mask=pad_mask, need_attn_weights=need_attn_weights)
        x = self.lm_mask_head(representation)

        result = {"logits": x, "representation": representation}
        if need_attn_weights:
            # result["attentions"] = torch.stack(attn_weights, dim=1)
            result["attentions"] = attn_weights
        return result
