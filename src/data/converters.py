import torch
from ..models.ernierna.converter import ErnieRNA_Converter
from .REtokenizer import ErnieBatchConverter

class ri_converter(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: list):
        descs, seqs = zip(*raw_batch)
        padded_toks = torch.tensor(self.alphabet.batch_tokenize(seqs))
        return descs, seqs, padded_toks


class SeqConverter(object):
    def __init__(self, model_type='fm', alphabet=None):
        self.model_type = model_type
        if (model_type == 'fm' and alphabet):
            self.alphabet = alphabet
            self.seqconverter = alphabet.get_batch_converter()
            self.pad_idx = alphabet.padding_idx
        elif 'rinalmo' in model_type:
            self.seqconverter = ri_converter(alphabet)
            self.pad_idx = alphabet.pad_idx
        elif model_type == 'ernie_rna':
            self.seqconverter = ErnieRNA_Converter()
            self.pad_idx = 1
        elif model_type == 'RNAErnie':
            self.seqconverter = ErnieBatchConverter()
            self.pad_idx = self.seqconverter.pad_idx
        else:
            raise NotImplementedError
    
    def __call__(self, raw_batch):
        if len(raw_batch) > 0 and isinstance(raw_batch[0], tuple) and len(raw_batch[0]) == 2 and isinstance(raw_batch[0][0], (tuple, list)):
            seq_pairs, fam_ids = zip(*raw_batch)
        else:
            seq_pairs = raw_batch
            fam_ids = None

        query_list = [('', seq_tuple[0][:1022]) for seq_tuple in seq_pairs]
        candidate_list = [('', seq_tuple[1][:1022]) for seq_tuple in seq_pairs]
        _, _, q_toks = self.seqconverter(query_list)
        _, _, c_toks = self.seqconverter(candidate_list)
        q_mask = q_toks==self.pad_idx
        c_mask = c_toks==self.pad_idx
        if fam_ids is None:
            return q_toks, c_toks, q_mask, c_mask
        fam_ids = torch.tensor(fam_ids, dtype=torch.long)
        return q_toks, c_toks, q_mask, c_mask, fam_ids


class SeqConverterWithSS(object):
    def __init__(self, alphabet, model_type='fm'):
        self.seqfamconverter = SeqConverter(alphabet=alphabet, model_type=model_type)

    def __call__(self, batch):
        seqbatch, ss_batch, fam_ids = zip(*batch)
        bsz = len(seqbatch)
        seqbatch = self.seqfamconverter(seqbatch)
        q_ss_tensors, c_ss_tensors = zip(*ss_batch)
        q_max_len = max([item.size(0) for item in q_ss_tensors]) + 2 # cls and eos
        c_max_len = max([item.size(0) for item in c_ss_tensors]) + 2
        q_ss = torch.zeros(bsz, q_max_len, q_max_len)
        c_ss = torch.zeros(bsz, c_max_len, c_max_len)
        for i, t in enumerate(q_ss_tensors):
            L = t.shape[0]
            q_ss[i, 1:L+1, 1:L+1] = t
        for i, t in enumerate(c_ss_tensors):
            L = t.shape[0]
            c_ss[i, 1:L+1, 1:L+1] = t
        fam_ids = torch.tensor(fam_ids, dtype=torch.long)
        return seqbatch, (q_ss, c_ss), fam_ids
    




##########################################################################################
#                                        Ali Converter                                   #
##########################################################################################
class AliConverter(SeqConverter):
    def __init__(self, model_type='fm', alphabet=None):
        super().__init__(model_type, alphabet)

    def __call__(self, raw_batch):
        seq_pairs, alignment_matrices = zip(*raw_batch)

        # Remove gaps for tokenisation
        query_list = [('', seq_tuple[0]) for seq_tuple in seq_pairs]
        candidate_list = [('', seq_tuple[1]) for seq_tuple in seq_pairs]
        _, _, q_toks = self.seqconverter(query_list)
        _, _, c_toks = self.seqconverter(candidate_list)
        q_mask = q_toks == self.pad_idx
        c_mask = c_toks == self.pad_idx

        B, N = q_toks.shape
        _, M = c_toks.shape

        N, M = N - 2, M - 2

        true_aln = torch.zeros(B, N, M)
        aln_mask = torch.zeros(B, N, M, dtype=torch.bool)

        for i, mat in enumerate(alignment_matrices):
            h, w = mat.shape
            true_aln[i, :h, :w] = mat
            v_q = len(seq_pairs[i][0])
            v_t = len(seq_pairs[i][1])
            aln_mask[i, :v_q, :v_t] = True

        return q_toks, c_toks, q_mask, c_mask, true_aln, aln_mask

