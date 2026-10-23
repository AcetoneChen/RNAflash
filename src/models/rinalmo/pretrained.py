import torch
from pathlib import Path
import json

from .alphabet import Alphabet
from .model import RiNALMo
from .config import model_config


def get_pretrained_model(pretrained_weights_path, lm_config: str = "giga") -> None:

    config = model_config(lm_config)
    model = RiNALMo(config)
    alphabet = Alphabet(**config['alphabet'])
    model.load_state_dict(torch.load(pretrained_weights_path, weights_only=False))

    return model, alphabet
