import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from itertools import chain
from gensim.models import KeyedVectors


### Very basic LSTM model for text classification

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_layer, hidden_dim, output_dim, n_layers, dropout, pos=False, pos_emb=None):
        super(LSTMClassifier, self).__init__()
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        emb_dim = emb_layer.weight.shape
        lstm_dim = hidden_dim
        self.pos = pos
        self.embedding = emb_layer
        self.lstm = nn.LSTM(emb_dim[1], hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
        
        if self.pos:
            self.pos_emb = pos_emb
            pos_emb_dim = pos_emb.weight.shape
            self.pos_lstm = nn.LSTM(pos_emb_dim[1], hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
            lstm_dim = hidden_dim*2

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_dim*2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pos_x=None):
        #print(x.shape)
        embedded = self.embedding(x)
        #print(embedded.shape)
        _, (hidden, _) = self.lstm(embedded)
        #print(hidden.shape)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        #print(hidden.shape)

        if self.pos:
            embedded_pos = self.pos_emb(pos_x)
            #print(embedded_pos.shape)
            _, (hidden_pos, _) = self.pos_lstm(embedded_pos)
            hidden_pos = torch.cat((hidden_pos[-2, :, :], hidden_pos[-1, :, :]), dim=1)
            #print(hidden_pos.shape)
            hidden = torch.concat([hidden, hidden_pos], dim=1)

        #print(hidden.shape)
        hidden = self.dropout(hidden)
        hidden = self.fc(hidden)
        #print(hidden.shape)
        return hidden.squeeze(1)
        #return self.sigmoid(hidden).squeeze(1)

