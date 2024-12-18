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
    def __init__(self, vocab_size, emb_layer, hidden_dim, output_dim, n_layers, dropout, pos_emb=None):
        super(LSTMClassifier, self).__init__()
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        emb_dim = emb_layer.weight.shape
        lstm_dim = hidden_dim
        self.embedding = emb_layer
        self.lstm = nn.LSTM(emb_dim[1], hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
        if pos_emb:
            self.pos_emb = pos_emb
            pos_emb_dim = pos_emb.weight.shape
            self.pos_lstm = nn.LSTM(pos_emb_dim[1], hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout, bidirectional=True)
            lstm_dim = hidden_dim*2

        self.fc = nn.Linear(lstm_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pos_x=None):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = hidden[-1]  # Take the output of the last LSTM layer

        if pos_x:
            embedded_pos = self.pos_emb(x)
            _, (hidden_pos, _) = self.pos_lstm(embedded)
            hidden_pos = hidden_pos[-1]
            out = torch.concat([out, hidden_pos], dim=0)

        out = self.fc(out)
        return self.sigmoid(out)

