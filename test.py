import json 
import numpy as np
from models import Tokenizer
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from pprint import pprint


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('text', action='store', type=str)
args = parser.parse_args()
print(args.text)