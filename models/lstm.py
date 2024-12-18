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
    def __init__(
        self,
        vocab_size,
        emb_layer,
        hidden_dim,
        output_dim,
        n_layers,
        dropout,
        pos_emb=None,
    ):
        super(LSTMClassifier, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        emb_dim = emb_layer.weight.shape
        lstm_dim = hidden_dim
        self.embedding = emb_layer
        self.lstm = nn.LSTM(
            emb_dim[1],
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        if pos_emb:
            self.pos_emb = pos_emb
            pos_emb_dim = pos_emb.weight.shape
            self.pos_lstm = nn.LSTM(
                pos_emb_dim[1],
                hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
            )
            lstm_dim = hidden_dim * 2

        self.fc = nn.Linear(lstm_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths, pos_x=None, pos_lengths=None):
        # Process the main input sequence
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, _) = self.lstm(packed_embedded)
        hidden = hidden[-1]  # Use the last hidden state from the final LSTM layer

        if self.pos_emb and pos_x is not None and pos_lengths is not None:
            # Process the positional embedding input
            embedded_pos = self.pos_emb(pos_x)  # [batch_size, seq_len, pos_emb_dim]
            packed_embedded_pos = nn.utils.rnn.pack_padded_sequence(
                embedded_pos, pos_lengths, batch_first=True, enforce_sorted=False
            )
            packed_output_pos, (hidden_pos, _) = self.pos_lstm(packed_embedded_pos)
            hidden_pos = hidden_pos[
                -1
            ]  # Use the last hidden state from the final pos_lstm layer

            # Concatenate outputs from both LSTMs
            hidden = torch.cat([hidden, hidden_pos], dim=1)

        # Fully connected layer
        output = self.fc(hidden)
        return self.sigmoid(output)
