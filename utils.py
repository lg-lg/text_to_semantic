from model.model import GPT, GPTConfig
import os
import torch
from transformers import BertTokenizer
import yaml
import torch.nn.functional as F


REMOTE_MODEL_PATHS = {
    "text_small": {
        "repo_id": "suno/bark",
        "file_name": "text.pt",
        "checksum": "b3e42bcbab23b688355cd44128c4cdd3",
    },
    "text": {
        "repo_id": "suno/bark",
        "file_name": "text_2.pt",
        "checksum": "54afa89d65e318d4f5f80e8e8799026a",
    },
}


def _clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_text_to_semantic_model(ckpt_path, device, use_small=False):
    ConfigClass = GPTConfig
    ModelClass = GPT

    model_key = f"text" if not use_small else f"text_small"

    if not os.path.exists(ckpt_path):
        print(f"text model not found, downloading into `{ckpt_path}`.")
        raise ValueError("text model not found")
    checkpoint = torch.load(ckpt_path, map_location=device)
    # this is a hack
    model_args = checkpoint["model_args"]
    if "input_vocab_size" not in model_args:
        model_args["input_vocab_size"] = model_args["vocab_size"]
        model_args["output_vocab_size"] = model_args["vocab_size"]
        del model_args["vocab_size"]
    gptconf = ConfigClass(**checkpoint["model_args"])
    model = ModelClass(gptconf)
    state_dict = checkpoint["model"]
    # fixup checkpoint
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    extra_keys = set(state_dict.keys()) - set(model.state_dict().keys())
    extra_keys = set([k for k in extra_keys if not k.endswith(".attn.bias")])
    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = set([k for k in missing_keys if not k.endswith(".attn.bias")])
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    model.load_state_dict(state_dict, strict=False)
    n_params = model.get_num_params()
    val_loss = checkpoint["best_val_loss"].item()
    print(f"Loaded text model with {n_params} params, val_loss={val_loss:.4f}.")
    del checkpoint, state_dict
    _clear_cuda_cache()
    tokenizer = BertTokenizer.from_pretrained("/home/shun.lei/.cache/huggingface/hub/models--bert-base-multilingual-cased")
    return model, tokenizer


def compute_loss(logits, target, mask):
    logits = logits.contiguous()
    target = target.contiguous().squeeze(-1)
    mask = mask.contiguous()

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)

    # top10
    _, top3 = torch.topk(log_probs_flat, 3, dim=-1)  # [B*T, 10]
    check_acc = (top3 == target_flat).any(dim=-1).float().view(*target.size()) * mask
    top3_acc = check_acc.sum() / mask.sum()

    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size()) * mask
    # mask: (batch, max_len)
    loss = losses.sum() / mask.sum()
    return loss, top3_acc


def cycle(dl):
    while True:
        for data in dl:
            yield data


def sequence_mask(seq_lens, max_len=None, device='cpu'):
    b = seq_lens.shape[0]
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device)  # [1, t]
    mask = mask < (seq_lens.unsqueeze(1))  # [1, t] + [b, 1] = [b, t]
    mask = mask.float()
    return mask
def calculate_model_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total params: {}, size of saving: {}M".format(
        pytorch_total_params,
        pytorch_total_params * 4 / 1024 / 1024))
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable params: {}, size of saving: {}M".format(
        pytorch_total_params,
        pytorch_total_params * 4 / 1024 / 1024))
    return

class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_config_from_file(file):
    with open(file, 'r') as f:
        hp = yaml.load(f, Loader=yaml.FullLoader)
    hp = HParams(**hp)
    return hp


def calculate_model_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total params: {}, size of saving: {}M".format(
        pytorch_total_params,
        pytorch_total_params * 4 / 1024 / 1024))
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable params: {}, size of saving: {}M".format(
        pytorch_total_params,
        pytorch_total_params * 4 / 1024 / 1024))
    return
def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log
def exists(val):
    return val is not None