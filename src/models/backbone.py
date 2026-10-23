from argparse import Namespace

from torch import nn


class Backbone(nn.Module):
    """Build the architectures used by the released RNAflash checkpoints.

    Construction does not read pretrained weights. RNAflashR/RNAflashA restore
    all parameters from their own checkpoint via load_from_checkpoint(). These
    explicit presets describe the released models, not arbitrary new variants.
    """

    def __init__(self, pretrained_model, bibert=False, freeze=False, *args, **kwargs):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.bibert = bibert
        self.freeze_bert = freeze
        if self.pretrained_model == 'fm':
            from .fm.alphabet import Alphabet
            from .fm.esm1 import BioBertModel

            # Architecture and behavior from the original RNA-FM model args.
            model_args = Namespace(
                arch='roberta_large',
                layers=12,
                embed_dim=640,
                ffn_embed_dim=5120,
                attention_heads=20,
                max_positions=1024,
                emb_layer_norm_before=True,
                token_dropout=False,
            )
            self.alphabet = Alphabet.from_architecture(model_args.arch, theme='rna')
            self.bert = BioBertModel(model_args, self.alphabet)
            self.bert_dim = model_args.embed_dim
            self.pad_idx = self.bert.padding_idx
        elif self.pretrained_model in ('rinalmo', 'rinalmo-mega', 'rinalmo-micro', 'rinalmo-giga'):
            from .rinalmo.alphabet import Alphabet
            from .rinalmo.config import model_config
            from .rinalmo.model import RiNALMo

            size = 'mega' if self.pretrained_model == 'rinalmo' else self.pretrained_model.split('-')[1]
            config = model_config(size)
            self.bert = RiNALMo(config)
            self.alphabet = Alphabet(**config['alphabet'])
            self.bert_dim = config.globals.embed_dim
            if size == 'giga':
                for block in self.bert.transformer.blocks[:27]:
                    for param in block.parameters():
                        param.requires_grad = False
            self.pad_idx = self.bert.pad_tkn_idx
        elif self.pretrained_model == 'ernie_rna':
            from .ernierna.model import ErnieRNAModel

            self.bert = ErnieRNAModel(
                vocab_size=25,
                embed_dim=768,
                num_layers=12,
                num_heads=12,
                ffn_dim=3072,
                max_seq_len=1024,
                dropout=0.1,
                attention_dropout=0.1,
                activation_dropout=0.0,
                num_segments=2,
                padding_idx=1,
                activation_fn='gelu',
                pooler_activation_fn='tanh',
                base_range=1,
                lamda=0.8,
            )
            self.bert_dim = self.bert.encoder.sentence_encoder.embedding_dim
            self.pad_idx = self.bert.padding_idx
            self.alphabet = None
        elif self.pretrained_model == 'RNAErnie':
            from transformers import BertConfig, BertModel

            # Preserve RNAErnie's config, including its non-default vocabulary,
            # position limit and layer-norm epsilon. Its MLM BERT has no pooler.
            config = BertConfig(
                vocab_size=11,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act='gelu',
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=2048,
                type_vocab_size=2,
                initializer_range=0.02,
                layer_norm_eps=1e-5,
                pad_token_id=0,
                position_embedding_type='absolute',
                use_cache=True,
                classifier_dropout=None,
            )
            self.bert = BertModel(config, add_pooling_layer=False)
            self.bert_dim = config.hidden_size
            self.pad_idx = config.pad_token_id
            self.bert.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.alphabet = None
        else:
            raise NotImplementedError(f'Unsupported backbone: {pretrained_model}')

        assert not (self.bibert and self.freeze_bert), "Do not use bi-bert if you don't finetune!"
        if self.bibert:
            raise NotImplementedError
        if self.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, toks, pad_mask, get_attn=False, use_ckpt=True):
        if self.freeze_bert:
            self.bert.eval()
        if self.pretrained_model == 'fm':
            info = self.bert(toks, repr_layers=[12], need_head_weights=get_attn, use_ckpt=use_ckpt)
            emb = info['representations'][12]
        elif 'rinalmo' in self.pretrained_model:
            info = self.bert(toks, need_attn_weights=get_attn, use_checkpoint=use_ckpt)
            emb = info['representation']
        elif self.pretrained_model == 'ernie_rna':
            if get_attn:
                emb, attn_map = self.bert(toks, use_checkpoint=use_ckpt, return_attn=True)
                return emb, attn_map
            emb = self.bert(toks, use_checkpoint=use_ckpt)
        elif self.pretrained_model == 'RNAErnie':
            emb = self.bert(toks, attention_mask=pad_mask).last_hidden_state
        return emb
    
