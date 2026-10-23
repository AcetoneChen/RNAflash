import argparse
import mmap
import os
from time import time

import faiss
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.converters import ri_converter
from src.models.ernierna.converter import ErnieRNA_Converter
from src.models.RNAflashR import RNAflashR

device = torch.device('cuda:2')

USE_MMAP = True


INDEX = 'datasets/RNAcentral_27/faiss/faiss_index_fp32.index'
FASTA_IDX = 'datasets/RNAcentral_27/rnacentral_active_1022_v27_idx.npy'
DATABASE_FASTA = 'datasets/RNAcentral_27/rnacentral_active_1022_v27.fasta'


class DatabaseConverter(object):
    def __init__(self, model_type='fm', alphabet=None):
        self.model_type = model_type
        if (model_type == 'fm' and alphabet):
            self.alphabet = alphabet
            self.seqconverter = alphabet.get_batch_converter()
            self.pad_idx = alphabet.padding_idx
        elif (model_type == 'rinalmo'):
            self.seqconverter = ri_converter(alphabet)
            self.pad_idx = alphabet.pad_idx
        elif (model_type == 'ernie_rna'):
            self.seqconverter = ErnieRNA_Converter()
            self.pad_idx = 1
        else:
            raise NotImplementedError

    def __call__(self, raw_batch):
        '''raw_batch: list((fam, seq))'''
        seq_list = [(fam, seq[:1022]) for fam, seq in raw_batch]
        fam_names, _, toks = self.seqconverter(seq_list)
        mask = toks == self.pad_idx
        return toks, mask, fam_names


def parse_fasta(fasta_path):
    with open(fasta_path, 'r') as f:
        for line in f:
            if line[0] != '>':
                return line.strip().upper().replace('T', 'U')


def search(index_path, search_target='RNAsolo_singlechain', probe=512,
           topk=1000):
    t_all = time()
    stats = {}
    L.seed_everything()

    ckpt_path = 'ckpts/RNAflashR.ckpt'

    # ---- embedding ----
    t = time()
    model = RNAflashR.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    targets = [t.split('.')[0] for t in os.listdir(f'datasets/{search_target}/fasta')]
    seqs = [('', parse_fasta(f'datasets/{search_target}/fasta/{t}.fasta'))
            for t in targets]
    converter = DatabaseConverter(model.backbone_k.pretrained_model,
                                  model.backbone_k.alphabet)
    loader = DataLoader(seqs, batch_size=16, collate_fn=converter, shuffle=False)

    print("Embedding queries:")
    embs = []
    with torch.no_grad():
        for batch in tqdm(loader):
            toks, pad_mask, _ = batch
            embs.append(model.embed(toks.to(device), pad_mask.to(device)).detach().cpu())
    query_embeds = np.ascontiguousarray(torch.vstack(embs).numpy().astype(np.float32))
    del model
    torch.cuda.empty_cache()
    stats['embed'] = time() - t

    # ---- load ----
    t = time()
    index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP if USE_MMAP else 0)
    index.nprobe = probe
    idx_map = np.load(FASTA_IDX)
    db_fd = os.open(DATABASE_FASTA, os.O_RDONLY)
    db = mmap.mmap(db_fd, 0, access=mmap.ACCESS_READ)
    stats['load'] = time() - t

    # ---- search ----
    t = time()
    D, I = index.search(query_embeds, topk)
    stats['search'] = time() - t

    # ---- write ----
    t = time()
    save_dir = f'datasets/{search_target}/search_all/search_nprobe{probe}_top{topk}'
    os.makedirs(save_dir, exist_ok=True)
    for i, target in enumerate(targets):
        with open(f'{save_dir}/{target}.fasta', 'w') as f:
            f.write(f'>query\n{seqs[i][1]}\n')
            for j, k in enumerate(I[i]):
                start, end = idx_map[k]
                desc, target_seq = db[start:end + 1].decode('utf-8').split('\n')
                f.write(f'{desc} | score: {D[i][j]}\n{target_seq.replace("T", "U")}\n')
    stats['write'] = time() - t

    db.close()
    os.close(db_fd)
    stats['total'] = time() - t_all
    return stats


def main(probe=512, topk=1000, search_target='RNAsolo_singlechain'):
    index_path = INDEX

    stats = search(index_path, search_target, probe, topk)

    W = 60
    print('\n' + '=' * W)
    print(f'Time for probe={probe}, top {topk}:')
    print('=' * W)
    print(f"search pipeline                       {stats['total']:9.2f}s")
    print(f"  ├ embedding                         {stats['embed']:9.2f}s")
    print(f"  ├ load                              {stats['load']:9.2f}s")
    print(f"  ├ index.search()                    {stats['search']:9.2f}s"
          f"   (nprobe={probe}, topk={topk})")
    print(f"  └ write                             {stats['write']:9.2f}s")
    print('=' * W)
    return stats


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', type=int, default=512)
    ap.add_argument('--topk', type=int, default=1000)
    ap.add_argument('--target', default='RNAsolo_singlechain')
    a = ap.parse_args()
    main(probe=a.probe, topk=a.topk, search_target=a.target)
