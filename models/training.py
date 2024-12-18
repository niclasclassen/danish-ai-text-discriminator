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
from embs import get_embs, get_pos_embs

# Tokenize text
def simple_tokenizer(text):
    return text.lower().split()

# Map words to integers
def tokenizer(text, vocab, unk_id):
    return [vocab.get(word, unk_id) for word in simple_tokenizer(text)]

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
    max_len = 2000

    data = data.sample(frac=1).reset_index(drop=True)
    X = data["Text"]
    y = data["label"]
    pdb.set_trace()
    train_texts, test_texts, train_labels, test_labels = train_test_split(X,y, test_size=0.2, random_state=42, stratify=data["label"])
    train_texts, val_texts, train_labels, val_labels = train_test_split(X,y, test_size=0.2, random_state=42, stratify=data["label"])

    train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, max_len, pad_id)
    val_dataset = TextDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, max_len, pad_id)
    test_dataset = TextDataset(test_texts.tolist(), test_labels.tolist(), tokenizer, max_len, pad_id)

    return train_dataset, val_dataset, test_dataset

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