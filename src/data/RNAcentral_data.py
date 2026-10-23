import array
import mmap
import os

import numpy as np
import torch
import torch.utils.data as tud
from torch.utils.data import IterableDataset, Dataset
import pickle
import subprocess

def fasta_iter(path):
    with open(path, 'r') as f:
        desc = None
        seq_chunks = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if desc is not None:
                    # t0 = time.time()
                    yield desc, ''.join(seq_chunks).upper().replace('T', 'U')
                    # t1 = time.time()
                    # print(f"CPU yield: {t1 - t0:.6f}s")
                desc = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if desc is not None:
            yield desc, ''.join(seq_chunks).upper().replace('T', 'U')


class FastaIterableDataset(IterableDataset):
    def __init__(self, fasta_path):
        self.fasta_path = fasta_path

    def __iter__(self):
        return fasta_iter(self.fasta_path)


class ShardedFastaDataset(IterableDataset):
    def __init__(self, fasta_path ):
        self.fasta_path  = fasta_path 

    def __iter__(self):
        info = tud.get_worker_info()
        if info is None:
            yield from fasta_iter(self.fasta_path)
            # raise AssertionError("info is None!")
        else:
            wid, nw = info.id, info.num_workers
            for i, item in enumerate(fasta_iter(self.fasta_path)):
                if i % nw == wid:
                    yield item



def build_fasta_index(fasta_path, index_path):
    fd = os.open(fasta_path, os.O_RDONLY)
    data = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    try:
        flat = array.array('q')
        pos, n = 0, len(data)
        while pos < n:
            nl = data.find(b'\n', pos)
            if data[pos] != 0x3E or nl < 0:  # 0x3E == '>'
                raise ValueError(f'offset {pos} 处不是合法的 FASTA 记录头')
            nl_seq = data.find(b'\n', nl + 1)
            end = nl_seq - 1 if nl_seq >= 0 else n - 1
            flat.append(pos)
            flat.append(end)
            if nl_seq < 0:
                break
            pos = end + 2
        index = np.frombuffer(flat, dtype=np.int64).reshape(-1, 2)
        np.save(index_path, index)
        return index
    finally:
        data.close()
        os.close(fd)


class IndexedFastaDataset(Dataset):
    def __init__(self, index_path, fasta_path):
        super().__init__()
        self.fasta_path = fasta_path
        self.index = np.load(index_path, mmap_mode='r')
        self.fd = None
        self.db = None

    def _get_db(self):
        if self.db is None:
            self.fd = os.open(self.fasta_path, os.O_RDONLY)
            self.db = mmap.mmap(self.fd, 0, access=mmap.ACCESS_READ)
        return self.db

    def __getitem__(self, index):
        db = self._get_db()
        start, end = self.index[index]
        desc, seq = db[start:end + 1].decode('utf-8').split('\n')
        seq = seq.upper().replace('T', 'U')
        return desc, seq

    def __len__(self):
        return len(self.index)
                    

# class IndexedFastaDataset(Dataset):
#     def __init__(self, index_path, fasta_path):
#         super().__init__()
#         self.fasta_path = fasta_path
#         with open(index_path, 'rb') as f:
#             self.index = pickle.load(f)
#         self.file = None

#     def _get_file(self):
#         if self.file is None:
#             self.file = open(self.fasta_path, "r")
#         return self.file
    
#     def __getitem__(self, index):
#         f = self._get_file()
#         f.seek(self.index[index][0])
#         content = f.read(self.index[index][1] - self.index[index][0] + 1)
#         desc, seq = content.split('\n')
#         seq = seq.upper().replace('T', 'U')
#         return desc, seq
    
#     def __len__(self):
#         return len(self.index)