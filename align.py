import lightning as L
import torch
import time

from torch.utils.data import DataLoader
import pickle, json
from tqdm import tqdm
import os

from src.models.RNAflashA import RNAflashA
from src.data.converters import ri_converter
from src.models.ernierna.converter import ErnieRNA_Converter



def parse_fasta(path):
    descs = []
    seqs = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip() and line[0] == '>':
                descs.append(line.strip()[1:])
            else:
                seqs.append(line.strip().upper().replace('T', 'U'))
    return descs, seqs


device = torch.device('cuda:2')

class DatabaseConverter(object):
    def __init__(self, model_type='fm', alphabet=None):
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
        raw_batch: list((fam, seq))
        '''
        seq_list = [(fam, seq[:1022]) for fam, seq in raw_batch]
        fam_names, _, toks = self.seqconverter(seq_list)
        mask = toks==self.pad_idx
        return toks, mask, fam_names
    

if __name__ == '__main__':
    start_time = time.time()

    ckpt_path = 'ckpts/RNAflashA.ckpt'
    model = RNAflashA.load_from_checkpoint(ckpt_path, 'cpu', weights_only=False)

    model.to(device)
    model.eval()

    fasta_dir = 'datasets/RNAsolo_singlechain/search_nprobe512_top1000_exp'

    a2m_dir = 'datasets/RNAsolo_singlechain/a2m_RNAflashA_nprobe512_top1000'


    os.makedirs(a2m_dir, exist_ok=True)
    converter = DatabaseConverter(
        model.backbone.pretrained_model, model.backbone.alphabet
    )
    fasta_names = os.listdir(fasta_dir)
    for i, fasta_name in enumerate(tqdm(fasta_names)):
        entity = fasta_name.split('.')[0]
        fasta_path = f'{fasta_dir}/{fasta_name}'
        a2m_path = f'{a2m_dir}/{entity}.a2m'
        
        descs, seqs = parse_fasta(fasta_path)
        name_seqs = list(zip(descs, seqs))
        _, query_seq = name_seqs[0]
        target_name_seqs = name_seqs[1:]
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            reranked_targets = model.reranking_ali_filter(query_seq, target_name_seqs, converter, bsz=8)
        with open(a2m_path, 'w') as f:
            f.write(f'>query\n{query_seq}\n')
            for name, gapped_seq, prob in reranked_targets:
                f.write(f'>{name} | {prob}\n{gapped_seq}\n')

    end_time = time.time()
    print(f'Time elapsed: {end_time-start_time}s.')
