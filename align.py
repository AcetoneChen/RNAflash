import argparse
import torch
import time

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


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def parse_args():
    parser = argparse.ArgumentParser(description='Rerank and align FAISS search results with RNAflashA.')
    parser.add_argument('--probe', type=positive_int, default=512)
    parser.add_argument('--topk', type=positive_int, default=1000)
    parser.add_argument('--target', default='RNAsolo_singlechain')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--checkpoint', default='ckpts/RNAflashA.ckpt')
    parser.add_argument('--batch-size', type=positive_int, default=8)
    parser.add_argument('--fasta-dir', help='Override the input directory derived from target, probe and topk')
    parser.add_argument('--output-dir', help='Override the alignment output directory')
    return parser.parse_args()

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
    

def main():
    args = parse_args()
    start_time = time.time()

    fasta_dir = args.fasta_dir or (
        f'datasets/{args.target}/search_all/search_nprobe{args.probe}_top{args.topk}'
    )
    a2m_dir = args.output_dir or (
        f'datasets/{args.target}/a2m_RNAflashA_nprobe{args.probe}_top{args.topk}'
    )
    fasta_names = sorted(
        name for name in os.listdir(fasta_dir)
        if name.endswith('.fasta') and os.path.isfile(os.path.join(fasta_dir, name))
    )
    if not fasta_names:
        raise ValueError(f'No .fasta files found in {fasta_dir}')

    device = torch.device(args.device)
    model = RNAflashA.load_from_checkpoint(args.checkpoint, 'cpu', weights_only=False)

    model.to(device)
    model.eval()

    os.makedirs(a2m_dir, exist_ok=True)
    converter = DatabaseConverter(
        model.backbone.pretrained_model, model.backbone.alphabet
    )
    for fasta_name in tqdm(fasta_names):
        entity = os.path.splitext(fasta_name)[0]
        fasta_path = f'{fasta_dir}/{fasta_name}'
        a2m_path = f'{a2m_dir}/{entity}.a2m'
        
        descs, seqs = parse_fasta(fasta_path)
        name_seqs = list(zip(descs, seqs))
        _, query_seq = name_seqs[0]
        target_name_seqs = name_seqs[1:]
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            reranked_targets = model.reranking_ali_filter(
                query_seq, target_name_seqs, converter, bsz=args.batch_size
            )
        with open(a2m_path, 'w') as f:
            f.write(f'>query\n{query_seq}\n')
            for name, gapped_seq, prob in reranked_targets:
                f.write(f'>{name} | {prob}\n{gapped_seq}\n')

    end_time = time.time()
    print(f'Time elapsed: {end_time-start_time}s.')


if __name__ == '__main__':
    main()
