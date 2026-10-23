"""
RNA sequence converter — converts (description, sequence) pairs into token tensors
ready for ErnieRNAModel.forward().

Usage:
    from src2.data.converter import convert_sequences

    pairs = [("seq1", "AUGCAUGCAUGC"), ("seq2", "GGGAAACCC")]
    names, seqs, tokens = convert_sequences(pairs)
    # tokens: [B, max_len+2] long tensor with CLS=0, PAD=1, EOS=2
"""

import torch
import numpy as np
from typing import List, Tuple


# Token index mapping
_TOKEN_MAP = {
    "G": 4, "g": 4,
    "A": 5, "a": 5,
    "U": 6, "T": 6, "t": 6, "u": 6,
    "C": 7, "c": 7,
}

CLS_IDX = 0
PAD_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def _seq_to_indices(sequence: str) -> np.ndarray:
    """Convert a single RNA sequence string to token indices (no CLS/EOS)."""
    indices = np.full(len(sequence), UNK_IDX, dtype=np.int64)
    for i, ch in enumerate(sequence):
        indices[i] = _TOKEN_MAP.get(ch, UNK_IDX)
    return indices


def get_pad_mask(tokens: torch.Tensor) -> torch.Tensor:
    """Compute boolean padding mask from token tensor.

    Args:
        tokens: [B, L] token tensor.

    Returns:
        [B, L] bool tensor, True where PAD_IDX.
    """
    return tokens.eq(PAD_IDX)


class ErnieRNA_Converter(object):
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        pairs: List[Tuple[str, str]],
        max_len: int = 1022,
    ) -> Tuple[List[str], List[str], torch.Tensor]:
        """Convert (description, RNA sequence) pairs to model-ready tokens.

        Each sequence gets wrapped with CLS (index 0) at the start and EOS (index 2)
        at the end. Sequences longer than ``max_len`` are truncated. Shorter
        sequences are padded with PAD (index 1).

        Args:
            pairs: list of (description, rna_sequence) tuples.
            max_len: maximum RNA sequence length (excluding CLS/EOS). Default 1022.

        Returns:
            (descriptions, sequences, tokens) where:
            - descriptions: list[str]
            - sequences: list[str] (original sequences, possibly truncated)
            - tokens: [B, L] torch.LongTensor, L = max_len_in_batch + 2
        """
        descriptions = []
        sequences = []

        # Truncate and collect
        for desc, seq in pairs:
            descriptions.append(desc)
            sequences.append(seq[:max_len])

        lengths = [len(s) for s in sequences]
        max_seq_len = max(lengths) if lengths else 0
        L = max_seq_len + 2  # +CLS +EOS

        B = len(pairs)
        tokens = np.full((B, L), PAD_IDX, dtype=np.int64)
        tokens[:, 0] = CLS_IDX

        for i, (seq, seq_len) in enumerate(zip(sequences, lengths)):
            tokens[i, 1:seq_len + 1] = _seq_to_indices(seq)
            tokens[i, seq_len + 1] = EOS_IDX

        return descriptions, sequences, torch.from_numpy(tokens)
