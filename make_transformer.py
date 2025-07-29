"""you give this script some words (one per line) and it will generate more things like it.
uses super state of the art Transformer AI tech
this code is intended to be super hackable.""" 

import os 
import sys 
import time 
import math 
import argparse 
from dataclasses import dataclass 
from typing import List 

import torch 
import torch.nn as nn 
from torch.nn import functional as F 
from torch.utils.data import Dataset 
from torch.utils.data.dataloader import DataLoader 
from torch.utils.tensorboard import SummaryWriter 

# ----------------------------------------- 

@dataclass 
class ModelConfig:
    block_size: int = None # length of the input sequences of integers 
    vocab_size: int = None # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently 
    n_layer: int = 4
    n_embd: int = 64 
    n_embd2: int = 64 
    n_head: int = 4 
    
# --------------------------------------------------------------
# Transformer Language Model (*exactly* as used in GPT-2) 


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper:  https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 
        # key, query, value projections for all heads, but in a batch 
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence 
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size)
                             )
        self.n_head = config.n_head 
        self.n_embd = config.n_embd 
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        