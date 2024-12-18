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
    def __init__(self, texts, labels, tokenizer, max_len, pad_id):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        tokens = self.tokenizer(text)
        length = len(tokens)

        padded_tokens = (
            tokens + [self.pad_id] * (self.max_len - len(tokens))
            if len(tokens) < self.max_len
            else tokens[: self.max_len]
        )

        return {
            "input": torch.tensor(padded_tokens, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
        }


def main():
    pass


if __name__ == "__main__":
    main()
