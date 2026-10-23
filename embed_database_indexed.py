import torch
from torch.utils.data import DataLoader, Subset
import faiss
import numpy as np
import math
import os
from tqdm import tqdm
import time
import json
import lightning as L
from src.data.RNAcentral_data import IndexedFastaDataset, build_fasta_index
from src.models.RNAflashR import RNAflashR
from src.models.ernierna.converter import ErnieRNA_Converter
from src.data.converters import ri_converter

#####################################
# config
#####################################

FASTA_PATH = "datasets/RNAcentral_27/rnacentral_active_1022_v27.fasta"
FASTA_INDEX_PATH = "datasets/RNAcentral_27/rnacentral_active_1022_v27_idx.npy"
BATCH_SIZE = 32
EMB_DIM = 256
NUM_WORKERS = 0
device = torch.device('cuda:2')

SAVE_DIR = f"datasets/RNAcentral_27/faiss"
EMB_SAVE_PATH      = f"{SAVE_DIR}/embeddings_fp16.npy"
FAISS_INDEX_PATH   = f"{SAVE_DIR}/faiss_index_fp32.index"
CHECKPOINT_PATH    = f"{SAVE_DIR}/embed_checkpoint.json"
RESERVOIR_PATH     = f"{SAVE_DIR}/reservoir.npz"

CHECKPOINT_EVERY = 5000  # save every 5000 batches
TRAIN_SIZE = 1_000_000
NLIST = 16384  # IVF cluster number

#####################################
# collate
#####################################

class DatabaseConverter(object):
    def __init__(self, model_type, alphabet=None):
        self.model_type = model_type
        if (model_type == 'fm' and alphabet):
            self.alphabet = alphabet
            self.seqconverter = alphabet.get_batch_converter()
            self.pad_idx = alphabet.padding_idx
        elif model_type == 'rinalmo':
            self.seqconverter = ri_converter(alphabet)
            self.pad_idx = alphabet.pad_idx
        elif model_type == 'ernie_rna':
            self.seqconverter = ErnieRNA_Converter()
            self.pad_idx = 1
        else:
            raise NotImplementedError

    def __call__(self, raw_batch):
        '''
        raw_batch: list((desc, seq))
        '''
        seq_list = [(desc, seq[:1022]) for desc, seq in raw_batch]
        desc, seqs, toks = self.seqconverter(seq_list)
        mask = toks == self.pad_idx
        return toks, mask, desc, seqs


#####################################
# vectorized reservoir sampling
#####################################

def reservoir_sample_batch(train_samples, emb_np, seen, train_size):
    """
    vectorize reservoir sampling

    for i-th seq in a batch (global ind: seen+i), the probability of being chosen to reservoir is:
    train_size / (seen+i+1)

    Param:
        train_samples: list[np.ndarray], current reservoir pooling
        emb_np: np.ndarray, shape (B, D), emb of current batch
        seen: int
        train_size: int, reservoir capacity

    Return:
        (train_samples, seen)
    """
    B = emb_np.shape[0]

    if len(train_samples) < train_size:
        # ---- stage 1：reservoir not full, fill directly ----
        remaining = train_size - len(train_samples)
        take = min(B, remaining)
        for i in range(take):
            train_samples.append(emb_np[i])

        if take < B:
            seen_start = seen + take
            remaining_emb = emb_np[take:]
            remaining_global = np.arange(seen_start, seen_start + len(remaining_emb))
            prob = train_size / (remaining_global + 1).astype(np.float64)
            selected = np.random.random(len(remaining_emb)) < prob
            if selected.any():
                replace_indices = np.random.randint(0, train_size, size=selected.sum())
                for k, idx in enumerate(replace_indices):
                    train_samples[idx] = remaining_emb[selected][k]

        seen += B
        return train_samples, seen

    # ---- stage 2: reservoir is full, substitute by probability ----
    seen_arr = np.arange(seen, seen + B)
    prob = train_size / (seen_arr + 1).astype(np.float64)
    selected = np.random.random(B) < prob

    if selected.any():
        replace_indices = np.random.randint(0, train_size, size=selected.sum())
        selected_emb = emb_np[selected]
        for k, idx in enumerate(replace_indices):
            train_samples[idx] = selected_emb[k]

    seen += B
    return train_samples, seen


#####################################
# checkpoint save / load
#####################################

def save_checkpoint(seen, next_id, train_samples):
    meta = {"seen": int(seen), "next_id": int(next_id)}
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(meta, f)
    os.replace(tmp, CHECKPOINT_PATH)

    arr = np.stack(train_samples).astype(np.float32) if len(train_samples) > 0 \
        else np.empty((0, EMB_DIM), dtype=np.float32)
    np.savez_compressed(RESERVOIR_PATH, samples=arr, seen=np.int64(seen))

    print(f"  [Checkpoint] seen={seen}, next_id={next_id}, reservoir={len(train_samples)}")


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(RESERVOIR_PATH):
        return None

    with open(CHECKPOINT_PATH, 'r') as f:
        meta = json.load(f)
    data = np.load(RESERVOIR_PATH)
    samples_arr = data["samples"]
    samples_list = [samples_arr[i] for i in range(samples_arr.shape[0])]

    print(f"  [Checkpoint] loaded: seen={meta['seen']}, next_id={meta['next_id']}, reservoir={len(samples_list)}")
    return meta["seen"], meta["next_id"], samples_list


def clear_checkpoint():
    for p in [CHECKPOINT_PATH, RESERVOIR_PATH]:
        if os.path.exists(p):
            os.remove(p)


#####################################
# main
#####################################

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # =========================
    # 0. build index (if needed)
    # =========================
    if not os.path.exists(FASTA_INDEX_PATH) or os.path.getsize(FASTA_INDEX_PATH) == 0:
        build_fasta_index(FASTA_PATH, FASTA_INDEX_PATH)

    # =========================
    # 1. build model
    # =========================
    L.seed_everything()
    ckpt_path = 'ckpts/RNAflashR.ckpt'

    model = RNAflashR.load_from_checkpoint(ckpt_path, 'cpu', weights_only=False)
    alphabet = model.backbone_k.alphabet
    model.to(device)
    model.eval()

    # =========================
    # 2. prepare data + resume checkpoint
    # =========================
    dataset = IndexedFastaDataset(FASTA_INDEX_PATH, FASTA_PATH)
    TOTAL_SEQS = len(dataset)

    ckpt = load_checkpoint()
    if ckpt is not None:
        seen, next_id, train_samples = ckpt
        emb_storage = np.memmap(EMB_SAVE_PATH, dtype=np.float16, mode='r+',
                                shape=(TOTAL_SEQS, EMB_DIM))
    else:
        seen, next_id, train_samples = 0, 0, []
        emb_storage = np.memmap(EMB_SAVE_PATH, dtype=np.float16, mode='w+',
                                shape=(TOTAL_SEQS, EMB_DIM))

    remaining_dataset = Subset(dataset, range(next_id, TOTAL_SEQS))
    remaining_batches = math.ceil(len(remaining_dataset) / BATCH_SIZE)

    collate_fn = DatabaseConverter(model.backbone_k.pretrained_model, alphabet)
    loader = DataLoader(remaining_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                        collate_fn=collate_fn, pin_memory=True, shuffle=False)

    # =========================
    # 3. Pass 1: encoding + reservoir sampling
    # =========================
    print(f"=== Pass 1: Encoding + Sampling (resume from next_id={next_id}) ===")
    print(f"    TOTAL_SEQS={TOTAL_SEQS}, remaining={len(remaining_dataset)}, remaining_batches={remaining_batches}")

    for step, (toks, pad_mask, desc, seqs) in enumerate(
            tqdm(loader, desc="Encoding")):

        toks = toks.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True)

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            emb = model.embed(toks, pad_mask)

        emb_np = emb.detach().cpu().numpy()

        # write memmap
        emb_storage[next_id:next_id + emb_np.shape[0], :] = emb_np.astype(np.float16)

        # vectorize reservoir sampling
        train_samples, seen = reservoir_sample_batch(train_samples, emb_np, seen, TRAIN_SIZE)

        next_id += emb_np.shape[0]

        # checkpoint
        if (step + 1) % CHECKPOINT_EVERY == 0:
            emb_storage.flush()
            save_checkpoint(seen, next_id, train_samples)

    # Pass 1 final checkpoint
    emb_storage.flush()
    save_checkpoint(seen, next_id, train_samples)

    if next_id != TOTAL_SEQS:
        print(f"WARNING: TOTAL_SEQS does not match\n  expected: {TOTAL_SEQS}\n  actual:   {next_id}")


    # =========================
    # 5. train IVF cluster
    # =========================
    print("=== Training FAISS IVF index ===")
    quantizer = faiss.IndexFlatIP(EMB_DIM)
    index = faiss.IndexIVFFlat(quantizer, EMB_DIM, NLIST, faiss.METRIC_INNER_PRODUCT)
    train_data = np.stack(train_samples).astype(np.float32)
    index.train(train_data)
    assert index.is_trained, "FAISS index training failed!"
    print("=== Training done ===")

    # =========================
    # 6. building index from memmap
    # =========================
    print("=== Building FAISS index from memmap ===")
    ADD_BATCH = 10000
    for i in tqdm(range(0, TOTAL_SEQS, ADD_BATCH), desc="Adding to index"):
        vecs = emb_storage[i:i + ADD_BATCH].astype(np.float32)
        index.add(vecs)

    index.nprobe = 32
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
