import ml_collections as mlc
import copy
from typing import List

# Tokens
ADENINE_TKN  = 'A'
CYTOSINE_TKN = 'C'
URACIL_TKN   = 'U'
GUANINE_TKN  = 'G'
THYMINE_TKN  = 'T'
INOSINE_TKN  = 'I'

ANY_NUCLEOTIDE_TKN = 'N'

# Tokens from https://en.wikipedia.org/wiki/FASTA_format
RNA_TOKENS = [ADENINE_TKN, CYTOSINE_TKN, GUANINE_TKN, THYMINE_TKN, INOSINE_TKN, "R", "Y", "K", "M", "S", "W", "B", "D", "H", "V", ANY_NUCLEOTIDE_TKN, "-"]

CLS_TKN  = "<cls>"
PAD_TKN  = "<pad>"
BOS_TKN  = "<bos>"
EOS_TKN  = "<eos>"
UNK_TKN  = "<unk>"
MASK_TKN = "<mask>"


class Alphabet:
    def __init__(
        self,
        standard_tkns: List[str] = RNA_TOKENS,
        special_tkns: List[str] = [CLS_TKN, PAD_TKN, EOS_TKN, UNK_TKN, MASK_TKN],
    ):
        super().__init__()

        self.standard_tkns = standard_tkns
        self.special_tkns = special_tkns

        self.idx_to_tkn = special_tkns + standard_tkns
        self.tkn_to_idx = {t : i for i, t in enumerate(self.idx_to_tkn)}

        self.cls_idx = self.tkn_to_idx[CLS_TKN]
        self.eos_idx = self.tkn_to_idx[EOS_TKN]

        self.unk_idx = self.tkn_to_idx[UNK_TKN]
        self.pad_idx = self.tkn_to_idx[PAD_TKN]
        self.mask_idx = self.tkn_to_idx[MASK_TKN]

    def __len__(self):
        return len(self.idx_to_tkn)

    def get_tkn(self, idx):
        return self.idx_to_tkn[idx]

    def get_idx(self, tkn):
        return self.tkn_to_idx.get(tkn, self.unk_idx)

    def encode(self, seq: str, pad_to_len: int = -1):
        seq = seq.upper()
        # Replace uracil tokens with thymine tokens
        seq = seq.replace(URACIL_TKN, THYMINE_TKN)

        encoded_seq = [self.cls_idx] + [self.get_idx(tkn) for tkn in seq] + [self.eos_idx]

        if len(encoded_seq) < pad_to_len:
            encoded_seq = encoded_seq + (pad_to_len - len(encoded_seq)) * [self.pad_idx]

        return encoded_seq
    
    def batch_tokenize(self, seqs: List[str]):
        max_len = max(len(seq) for seq in seqs)
        max_len += 2 # CLS and EOS

        batch = []
        for seq in seqs:
            batch.append(self.encode(seq, pad_to_len=max_len))

        return batch

def model_config(name):
    c = copy.deepcopy(default_config)

    if name == "nano":
        c.globals.embed_dim = 320

        c.model.transformer.num_blocks = 6
        c.model.transformer.num_heads = 20
    elif name == "micro":
        c.globals.embed_dim = 480

        c.model.transformer.num_blocks = 12
        c.model.transformer.num_heads = 20
    elif name == "mega":
        c.globals.embed_dim = 640

        c.model.transformer.num_blocks = 30
        c.model.transformer.num_heads = 20
    elif name == "giga":
        c.globals.embed_dim = 1280

        c.model.transformer.num_blocks = 33
        c.model.transformer.num_heads = 20

        c.training.optimizer.lr = 5e-5
        c.training.lr_scheduler.cosine_decay.eta_min = 1e-5
    else:
        raise ValueError("Invalid configuration name!")
    
    assert not any_tokenizer_discrepancies(c), "Found discrepancies in tokenizer configuration!"

    return c

def any_tokenizer_discrepancies(config):
    alphabet = Alphabet(**config['alphabet'])

    if alphabet.get_idx(MASK_TKN) != config['globals'].mask_tkn_idx:
        return True
    
    if alphabet.get_idx(PAD_TKN) != config['globals'].pad_tkn_idx:
        return True
    
    if len(alphabet) != config['globals'].alphabet_size:
        return True
    
    return False

embed_dim = mlc.FieldReference(480, field_type=int)

default_alphabet = Alphabet()
alphabet_size = mlc.FieldReference(len(default_alphabet), field_type=int)
mask_tkn_idx = mlc.FieldReference(default_alphabet.get_idx(MASK_TKN), field_type=int)
pad_tkn_idx = mlc.FieldReference(default_alphabet.get_idx(PAD_TKN), field_type=int)

mask_ratio = mlc.FieldReference(0.15, field_type=float)
mask_tkn_prob = mlc.FieldReference(0.8, field_type=float)

default_config = mlc.ConfigDict(
    {
        "globals": {
            "embed_dim": embed_dim,
            "alphabet_size": alphabet_size,
            "mask_tkn_idx": mask_tkn_idx,
            "pad_tkn_idx": pad_tkn_idx,
            "mask_ratio": mask_ratio,
            "mask_tkn_prob": mask_tkn_prob,
        },
        "alphabet": {
            "standard_tkns": RNA_TOKENS,
            "special_tkns": [CLS_TKN, PAD_TKN, EOS_TKN, UNK_TKN, MASK_TKN],
        },
        "training": {
            "optimizer": {
                "lr": 1e-4,
                "weight_decay": 0.01,
            },
            "lr_scheduler": {
                "warm_up": {
                    "iters": 2000,
                },
                "cosine_decay": {
                    "T_max": 200_000,
                    "eta_min": 1e-5,
                },
            },
            "masking": {
                "bert_masking": {
                    "mask_ratio": mask_ratio,
                    "mask_tkn_prob": mask_tkn_prob,
                    "random_tkn_prob": 0.1,
                }
            },
        },
        "model": {
            "embedding": {
                "num_embeddings": alphabet_size,
                "embedding_dim": embed_dim,
                "padding_idx": pad_tkn_idx,
            },
            "token_dropout": {
                "active": True,
                "mask_ratio": mask_ratio,
                "mask_tkn_prob": mask_tkn_prob,
                "mask_tkn_idx": mask_tkn_idx,
                "pad_tkn_idx": pad_tkn_idx,
            },
            "transformer": {
                "embed_dim": embed_dim,
                "num_blocks": 12,
                "num_heads": 20,
                "use_rot_emb": True,
                "attn_qkv_bias": False,
                "attention_dropout": 0.1,
                "transition_dropout": 0.0,
                "residual_dropout": 0.1,
                "transition_factor": 4,
                "use_flash_attn": True,
            },
            "lm_mask_head": {
                "embed_dim": embed_dim,
                "alphabet_size": alphabet_size,
            }
        }
    }
)
