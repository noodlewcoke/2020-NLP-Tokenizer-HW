import os 
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import json 
import numpy as np
from models import Tokenizer
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence


class Dataset:
    def __init__(self, data, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.data = data
        if shuffle: np.random.shuffle(self.data)
        self.data_len = len(self.data)
        self.counter = 0

    def __iter__(self):
        return self
        
    def __next__(self):
        if self.counter < self.data_len:
            last = min(self.counter+self.batch_size, self.data_len)
            batch = self.data[self.counter:last]
            batch = np.array(sorted(batch, key=lambda d: len(d[0]), reverse=True))
            sentence_lengths = [len(i[0]) for i in batch]
            data_x = batch[:, 0]
            data_y = batch[:, 1]
            sentence_lengths = [len(i) for i in data_x]
            x = [torch.LongTensor(i) for i in data_x]
            y = [torch.LongTensor(i) for i in data_y]
            padded_x = pad_sequence(x).transpose(0, 1)
            padded_y = pad_sequence(y).transpose(0, 1)
            self.counter = last
            return padded_x, padded_y, sentence_lengths
        else:
            raise StopIteration()


data = np.load('data/en.wiki.dev.npy')
dataset = Dataset(data, 32, True)

model = Tokenizer(631, 32, 64, lr=1e-3)


for data_x, data_y, X_lengths in dataset:

    loss = model.update(data_x, data_y, X_lengths)
    print(loss)








