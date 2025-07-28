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