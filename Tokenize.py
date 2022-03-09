import json 
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F 
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from pprint import pprint

class Tokenizer(torch.nn.Module):

    def __init__(self, vocab_dim, embedding_dim, hidden_dim, lr=1e-3):

        super(Tokenizer, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding_layer = torch.nn.Embedding(vocab_dim, embedding_dim, padding_idx=0)
        self.bilstm = torch.nn.LSTM(embedding_dim, hidden_dim)

        self.fc = torch.nn.Linear(hidden_dim, 4)

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, sentence, X_lengths):
        x = sentence 
        x = self.embedding_layer(x)

        x = torch.nn.utils.rnn.pack_padded_sequence(x, X_lengths, batch_first=True)

        x, self.hidden = self.bilstm(x)        

        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        self.logits = self.fc(x)
        self.output = F.softmax(self.logits, dim=2)
        return self.logits, self.output

    def update(self, sentence, label, X_lengths):
        output, _ = self(sentence, X_lengths)

        output = output.view(-1, 4)

        label = label.contiguous()
        label = label.view(-1)

        self.optimizer.zero_grad()
        loss = self.loss_function(output, label)

        loss.backward()
        self.optimizer.step()
        return loss

    def evaluate(self, sentence, label, X_lenghts):
        with torch.no_grad():
            output, _ = self(sentence, X_lengths)

            output = output.view(-1, 4)

            label = label.contiguous()
            label = label.view(-1)

            self.optimizer.zero_grad()
            loss = self.loss_function(output, label)
        return loss

    def prediction(self, sentence, X_lengths):
        with torch.no_grad():
            _, output = self(sentence, X_lengths)
            return torch.argmax(output[-1][:, 1:], dim=1)+1

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location='cpu'))

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
    return ''.join(bis_sentence)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('text', action='store', type=str)
args = parser.parse_args()


filename = args.text


FNAMES = [filename+'.sentences.train',
            filename+'.sentences.dev',
        ]

vocabulary = []
for FNAME in FNAMES:
    with open(FNAME, 'r') as f:
        data = f.read().split('\n')
        f.close()
        textdata = ''.join(data)
        unique = list(set(textdata))
        vocabulary += unique

vocabulary = sorted(list(set(vocabulary)))
vocabulary.insert(0, '<PAD>')
vocabulary.append('<UNK>')
vocabulary_dict = {k:v for v, k in enumerate(vocabulary)}


FILENAMES = [
            filename+'.sentences.train',
            filename+'.sentences.dev',
            ]
GOLDNAMES = [
            filename+'.gold.train',
            filename+'.gold.dev',
            ]
TESTNAMES = [
            filename+'.sentences.test'
            ]
outputNames = [
                filename+'.train',
                filename+'.dev',
                filename+'.test'
]

c = 0
for s, g in zip(FILENAMES, GOLDNAMES):
    with open(s, 'r') as f:
        sentences = f.read().split('\n')
        f.close()
    with open(g, 'r') as f:
        labels = f.read().split('\n')
        f.close()
    dataset = []
    for sentence, label in zip(sentences, labels):
        if not sentence:
            continue
        assert len(sentence) == len(label), "A sentence-label pair doesn't hold."
        enumSentence = [vocabulary_dict[s] for s in sentence]
        enumLabels = [] 
        for l in label:
            if l == 'B':
                enumLabels.append(1)
            elif l == 'I':
                enumLabels.append(2)
            elif l == 'S':
                enumLabels.append(3)
        dataset.append((enumSentence, enumLabels))
    np.save(outputNames[c], dataset)
    c += 1

for t in TESTNAMES:
    with open(t, 'r') as f:
        sentences = f.read().split('\n')
        f.close()
    dataset = []
    for sentence in sentences:
        if not sentence:
            continue
        enumSentence = [] 
        for s in sentence:
            if s in vocabulary_dict.keys():
                enumSentence.append(vocabulary_dict[s])
            else:
                enumSentence.append(vocabulary_dict['<UNK>'])
        dataset.append(enumSentence)
    np.save(outputNames[c], dataset)
    c += 1

EPOCHS = 1

data = np.load(filename+'.train.npy',allow_pickle=True)
training_set = Dataset(data, 32, shuffle=True)
data = np.load(filename+'.dev.npy',allow_pickle=True)
dev_set = Dataset(data, 32, shuffle=True)
del data
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Tokenizer(len(vocabulary), 32, 64, lr=1e-3).to(device)

for epoch in range(EPOCHS):
    epoch_loss = 0
    episode = 1
    for data_x, data_y, X_lengths in training_set:
        loss = model.update(data_x.to(device), data_y.to(device), torch.LongTensor(X_lengths).to(device))
        epoch_loss += loss
        if not episode%10:
            print('\tEpoch: {:2d}\t Episode {:2d}\t avg loss = {:0.4f}'.format(epoch, episode, epoch_loss / episode))
        episode += 1
        break
    print('\tEpoch: {:2d}\t avg training loss = {:0.4f}'.format(epoch, epoch_loss / episode))

    dev_loss = 0
    episode = 1
    for data_x, data_y, X_lengths in dev_set:
        loss = model.evaluate(data_x.to(device), data_y.to(device), torch.LongTensor(X_lengths).to(device))
        dev_loss += loss
        episode += 1
        break
    print('\tEpoch: {:2d}\t avg dev loss = {:0.4f}'.format(epoch, dev_loss / episode))
    break

data = np.load(filename+'.test.npy')
dataset = Dataset(data, 1, shuffle=False, test=True)
output_file = []

for data_x, X_lengths in dataset:
    output = model.prediction(data_x.to(device), torch.LongTensor(X_lengths).to(device))
    prediction = pred2bis(output)
    output_file.append(prediction)
    print(prediction)

with open(filename+'.predicted.test', 'w') as f:
    f.write('\n'.join(output_file))
    f.close()
