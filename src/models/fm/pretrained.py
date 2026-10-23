# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .esm1 import BioBertModel
from .alphabet import Alphabet
import torch
from argparse import Namespace
import warnings

def has_emb_layer_norm_before(model_state):
    """Determine whether layer norm needs to be applied before the encoder"""
    return any(k.startswith("emb_layer_norm_before") for k, param in model_state.items())


def from_pretrained(model_location: str):
    model_data = torch.load(model_location, map_location='cpu', weights_only=False)
    alphabet = Alphabet.from_architecture(model_data["args"].arch, theme="rna")
    # upgrade state dict
    pra = lambda s: "".join(s.split("encoder_")[1:] if "encoder" in s else s)
    prs1 = lambda s: "".join(s.split("encoder.")[1:] if "encoder" in s else s)
    prs2 = lambda s: "".join(
        s.split("sentence_encoder.")[1:] if "sentence_encoder" in s else s
    )
    model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
    model_state = {prs1(prs2(arg[0])): arg[1] for arg in model_data["model"].items()}
    model_state["embed_tokens.weight"][alphabet.mask_idx].zero_()  # For token drop
    model_args["emb_layer_norm_before"] = has_emb_layer_norm_before(model_state)
    
    model = BioBertModel(
        Namespace(**model_args),
        alphabet,
    )

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
    error_msgs = []
    missing = (expected_keys - found_keys) - expected_missing
    if missing:
        error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
    unexpected = found_keys - expected_keys
    if unexpected:
        error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

    if error_msgs:
        raise RuntimeError(
            "Error(s) in loading state_dict for {}:\n\t{}".format(
                model.__class__.__name__, "\n\t".join(error_msgs)
            )
        )
    if expected_missing - found_keys:
        warnings.warn(
            "Regression weights not found, predicting contacts will not produce correct results."
        )

    model.load_state_dict(model_state, strict=True)
    return model, alphabet


