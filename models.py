import numpy as np
import torch
import torch.nn.functional as F 
from torch.autograd import Variable


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

    # def init_hidden(self):
    #     # Before we've done anything, we dont have any hidden state.
    #     # Refer to the Pytorch documentation to see exactly
    #     # why they have this dimensionality.
    #     # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
    #     return (Variable(torch.zeros(2, 5, self.hidden_dim)),   
    #             Variable(torch.zeros(2, 5, self.hidden_dim)))
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
            # print(output[-1][:, 1:])
            # return torch.argmax(output, dim=2)
            return torch.argmax(output[-1][:, 1:], dim=1)+1


        

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location='cpu'))



