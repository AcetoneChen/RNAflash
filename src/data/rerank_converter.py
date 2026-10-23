from .converters import SeqConverter
import torch

def group_mat(anchor_toks, target_toks, ali_mats, target_seqs):
    B, N = anchor_toks.shape
    _, M = target_toks.shape

    anchor_len = N - 2

    true_aln = torch.zeros(B, N, M) # cls and eos are included, which are not easy to remove, so we keep them as 0.
    aln_mask = torch.zeros(B, N, M, dtype=torch.bool)

    for i, mat in enumerate(ali_mats):
        h, w = mat.shape
        true_aln[i, 1:h+1, 1:w+1] = mat
        v_q = anchor_len
        v_t = len(target_seqs[i])
        aln_mask[i, 1:v_q+1, 1:v_t+1] = True
    return true_aln, aln_mask


class AliRerankConverter(SeqConverter):
    """
    The batch is actually organized in RerankingDataset. The collat_fn just pad them.
    """
    def __init__(self, model_type='fm', alphabet=None):
        super().__init__(model_type, alphabet)

    def __call__(self, batch):
        anchor, sampled_pos, sampled_neg, alignment_matrices = batch[0]
        
        pos_list = [('', seq) for seq in sampled_pos]
        neg_list = [('', seq) for seq in sampled_neg]

        _, _, anchor_toks = self.seqconverter([('', anchor)])
        anchor_toks = anchor_toks.repeat(len(pos_list), 1)
        anchor_mask = anchor_toks == self.pad_idx

        _, _, pos_toks = self.seqconverter(pos_list)
        pos_mask = pos_toks == self.pad_idx

        _, _, neg_toks = self.seqconverter(neg_list)
        neg_mask = neg_toks == self.pad_idx


        B, N = anchor_toks.shape
        _, M = pos_toks.shape

        N, M = N - 2, M - 2

        true_aln = torch.zeros(B, N+2, M+2) # +2 for cls and eos, which is not easy to remove, so we keep them as 0.
        aln_mask = torch.zeros(B, N+2, M+2, dtype=torch.bool)

        for i, mat in enumerate(alignment_matrices):
            h, w = mat.shape
            true_aln[i, 1:h+1, 1:w+1] = mat
            v_q = len(anchor)
            v_t = len(sampled_pos[i])
            aln_mask[i, 1:v_q+1, 1:v_t+1] = True

        return anchor_toks, pos_toks, neg_toks, \
               anchor_mask, pos_mask, neg_mask, \
               true_aln, aln_mask
    

class BiAliRerankConverter(AliRerankConverter):
    def __init__(self, model_type='fm', alphabet=None):
        super().__init__(model_type, alphabet)
    
    def __call__(self, batch):
        anchor, sampled_pos, sampled_neg, pos_aln_mats, neg_aln_mats = batch[0]
        
        pos_list = [('', seq) for seq in sampled_pos]
        neg_list = [('', seq) for seq in sampled_neg]

        _, _, anchor_toks = self.seqconverter([('', anchor)])
        anchor_toks = anchor_toks.repeat(len(pos_list), 1)
        anchor_mask = anchor_toks == self.pad_idx

        _, _, pos_toks = self.seqconverter(pos_list)
        pos_mask = pos_toks == self.pad_idx
        _, _, neg_toks = self.seqconverter(neg_list)
        neg_mask = neg_toks == self.pad_idx

        pos_true_aln, pos_aln_mask = group_mat(anchor_toks, pos_toks, pos_aln_mats, sampled_pos)
        neg_true_aln, neg_aln_mask = group_mat(anchor_toks, neg_toks, neg_aln_mats, sampled_neg)

        return anchor_toks, pos_toks, neg_toks, \
               anchor_mask, pos_mask, neg_mask, \
               pos_true_aln, pos_aln_mask, \
               neg_true_aln, neg_aln_mask
    

class CSSAliRerankConverter(AliRerankConverter):
    def __init__(self, model_type='fm', alphabet=None):
        super().__init__(model_type, alphabet)
    
    def __call__(self, batch):
        anchor, sampled_pos, sampled_neg, aln_mats, css_aln_mats = batch[0]
        pos_list = [('', seq) for seq in sampled_pos]
        neg_list = [('', seq) for seq in sampled_neg]

        _, _, anchor_toks = self.seqconverter([('', anchor)])
        anchor_toks = anchor_toks.repeat(len(pos_list), 1)
        anchor_mask = anchor_toks == self.pad_idx

        _, _, pos_toks = self.seqconverter(pos_list)
        pos_mask = pos_toks == self.pad_idx
        _, _, neg_toks = self.seqconverter(neg_list)
        neg_mask = neg_toks == self.pad_idx

        true_aln, aln_mask = group_mat(anchor_toks, pos_toks, aln_mats, sampled_pos)
        css_true_aln, _ = group_mat(anchor_toks, pos_toks, css_aln_mats, sampled_pos)

        return anchor_toks, pos_toks, neg_toks, \
               anchor_mask, pos_mask, neg_mask, \
               true_aln, css_true_aln, aln_mask
        