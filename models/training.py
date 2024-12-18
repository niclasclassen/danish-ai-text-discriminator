import sys
sys.path.append("..")
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
import pdb
from dataloading.dataset import TextDataset
from models.lstm import LSTMClassifier
from embs import get_embs, get_pos_embs

class Tokenizer:
    def __init__(self, vocab, unk_id, pos_vocab, pos_unk_id, pos_pad_id, max_len):
        self.vocab = vocab
        self.unk_id = unk_id
        self.pos_vocab = pos_vocab
        self.pos_unk_id = pos_unk_id
        self.pos_pad_id = pos_pad_id
        self.max_len = max_len

    # Tokenize text
    def simple_tokenizer(self, text):
        return text.lower().split()

    def tokenize(self, text):
        tokens = self.simple_tokenizer(text)
        word_ids = [0] + [self.vocab.get(word, self.unk_id) for word in tokens]
        word_ids = word_ids + [0 for _ in range(self.max_len - len(tokens) - 1)]
        pos_ids = [self.pos_pad_id] + [self.pos_vocab.get(word, self.pos_unk_id) for word in tokens]
        pos_ids = pos_ids + [self.pos_pad_id for _ in range(self.max_len - len(tokens) - 1)]
        return word_ids, pos_ids

def load_data():
    #data = pd.read_csv("../reddit.csv") # Load the data
    data = pd.read_csv("../cleaned_final.csv")
    human_df = data[["Combined Text"]]
    human_df["label"] = 0
    ai_df = data[["Rewritten Text"]]
    ai_df["label"] = 1
    human_df = human_df.rename(columns={"Combined Text": "Text"})
    ai_df = ai_df.rename(columns={"Rewritten Text": "Text"})
    data = pd.concat([human_df, ai_df], ignore_index=True)
    return data

def train():
    data = load_data()
    embedding, vocab, unk_id, pad_id = get_embs()
    pos_embedding, pos_vocab, pos_unk_id, pos_pad_id = get_pos_embs(data)
    max_len = 1000
    tokenizer = Tokenizer(vocab, unk_id, pos_vocab, pos_unk_id, pos_pad_id, max_len)

    data = data.sample(frac=1).reset_index(drop=True)
    train_texts, test_texts, train_labels, test_labels = train_test_split(data["Text"], data["label"], test_size=0.2, random_state=42, stratify=data["label"])    
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

    train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
    val_dataset = TextDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)
    test_dataset = TextDataset(test_texts.tolist(), test_labels.tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #vocab_size = len(vocab) + 1  # Add 1 for padding index
    vocab_size = len(vocab)
    embedding_dim = 128
    hidden_dim = 64
    output_dim = 1
    n_layers = 2
    dropout = 0.5

    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        for batch in train_loader:
            inputs = batch['input']
            labels = batch['label']

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')


    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input']
            labels = batch['label']
            outputs = model(inputs).squeeze()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(f'Test Accuracy: {correct / total * 100:.2f}%')

def main():
    train()

if __name__ == "__main__":
    main()