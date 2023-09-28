import os
from pathlib import Path

import torch

from hubert.hubert_base import CustomHubert
from hubert.tokenizer import CustomTokenizer

from torch import nn

import logging

logging.root.setLevel(logging.ERROR)
import sys


def exists(val):
    return val is not None


class HubertWithKmeans(nn.Module):
    """
    checkpoint and kmeans can be downloaded at https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
    or you can train your own
    """

    def __init__(
            self,
            hubert_path,
            kmeans_path,
            target_sample_hz=16000,
            device = 'cpu',
            seq_len_multiple_of=None,
    ):
        super().__init__()
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = seq_len_multiple_of

        hubert_path = Path(hubert_path)
        kmeans_path = Path(kmeans_path)

        assert hubert_path.exists(), f'path {hubert_path} does not exist'
        assert kmeans_path.exists(), f'path {kmeans_path} does not exist'

        self.hubert = CustomHubert(checkpoint_path=str(hubert_path), target_sample_hz=target_sample_hz,seq_len_multiple_of=seq_len_multiple_of,device=device)
        self.hubert.eval()
        for param in self.hubert.parameters():
            param.requires_grad = False
        
        self.tokenizer = CustomTokenizer.load_from_checkpoint(kmeans_path).to(device)  
        self.tokenizer.eval()

        
    @torch.no_grad()
    def forward(
            self,
            wav_input,
            sr,
    ):
        device = wav_input.device

        semantic_embeds = self.hubert(wav_input,input_sample_hz = sr)
        semantic_tokens = self.tokenizer.get_token(semantic_embeds)
        semantic_tokens = semantic_tokens

        return semantic_tokens
