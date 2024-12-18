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

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        tokens, pos_tokens = self.tokenizer.tokenize(text)

        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'pos_tokens': torch.tensor(pos_tokens, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float),
        }


def main():
    pass

if __name__ == "__main__":
    main()