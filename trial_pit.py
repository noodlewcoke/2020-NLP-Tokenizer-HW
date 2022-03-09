import json 
import numpy as np
from models import Tokenizer
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from pprint import pprint

class Dataset:
    def __init__(self, data, batch_size, shuffle=False, test=False):
        self.batch_size = batch_size
        self.data = data
        if shuffle: np.random.shuffle(self.data)
        self.data_len = len(self.data)
        self.counter = 0
        self.test = test

    def __iter__(self):
        return self
        
    def __next__(self):
        if not self.test:
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
        else:
            if self.counter < self.data_len:
                last = min(self.counter+self.batch_size, self.data_len)
                batch = self.data[self.counter:last]
                batch = np.array(sorted(batch, key=lambda d: len(d), reverse=True))
                sentence_lengths = [len(i) for i in batch]
                data_x = batch[:]
                x = [torch.LongTensor(i) for i in data_x]
                padded_x = pad_sequence(x).transpose(0, 1)
                self.counter = last
                return padded_x, sentence_lengths
            else:
                raise StopIteration()

def pred2bis(sentence):
    bis_sentence = []
    for i in sentence:
        if i == 1:
            bis_sentence.append('B')
        elif i == 2:
            bis_sentence.append('I')
        elif i == 3:
            bis_sentence.append('S')
        else:
            print('problem here')
            bis_sentence.append('B')
    return ''.join(bis_sentence)

data = np.load('data/en.wiki.dev.npy')
dataset = Dataset(data, 1, shuffle=False, test=False)
model = Tokenizer(632, 32, 64, lr=1e-3)
model.load('experiments/exp2/tokenizer_model')
output_file = []
counter = 0
for data_x, data_y,X_lengths in dataset:
    output = model.prediction(data_x, X_lengths)
    print(output)
    print(data_y)
    prediction = pred2bis(output)
    output_file.append(prediction)
    print(counter)
    counter += 1
    break
