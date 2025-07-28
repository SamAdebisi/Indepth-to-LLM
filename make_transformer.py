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
# TRansformer Language Model (*exactly* as used in GPT-2)