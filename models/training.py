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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("IS CUDA?", torch.cuda.is_available(), flush=True)

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
        tokens = tokens[:self.max_len]
        word_ids = [0] + [self.vocab.get(word, self.unk_id) for word in tokens]
        word_ids = word_ids + [0 for _ in range(self.max_len - len(tokens) - 1)]
        pos_ids = [self.pos_pad_id] + [self.pos_vocab.get(word, self.pos_unk_id) for word in tokens]
        pos_ids = pos_ids + [self.pos_pad_id for _ in range(self.max_len - len(tokens) - 1)]
        return word_ids, pos_ids

def load_data():
    #data = pd.read_csv("../cleaned_final.csv")
    #data = pd.read_csv("../cleaned_final_title_based.csv")
    data = pd.read_csv("../combined_file_final.csv", sep=";")
    human_df = data[["Combined Text"]]
    human_df["label"] = 0
    ai_df = data[["Rewritten Text"]]
    ai_df["label"] = 1
    ai_df2 = data[["Generated Text"]]
    ai_df2["label"] = 1
    human_df = human_df.rename(columns={"Combined Text": "Text"})
    ai_df = ai_df.rename(columns={"Rewritten Text": "Text"})
    ai_df2 = ai_df2.rename(columns={"Generated Text": "Text"})
    data = pd.concat([human_df, ai_df, ai_df2], ignore_index=True)
    return data

def train(pos_emb, mode):
    data = load_data()
    embedding, vocab, unk_id = get_embs()
    pos_embedding, pos_vocab, pos_unk_id, pos_pad_id = get_pos_embs(data)
    max_len = 2000
    tokenizer = Tokenizer(vocab, unk_id, pos_vocab, pos_unk_id, pos_pad_id, max_len)

    data = data.sample(frac=1).reset_index(drop=True)
    train_texts, test_texts, train_labels, test_labels = train_test_split(data["Text"], data["label"], test_size=0.2, random_state=42, stratify=data["label"])    
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

    train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
    val_dataset = TextDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)
    test_dataset = TextDataset(test_texts.tolist(), test_labels.tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #vocab_size = len(vocab) + 1  # Add 1 for padding index
    vocab_size = len(vocab)
    #embedding_dim = 128

    output_dim = 1
    n_layers = 1
    dropout = 0.3
    #criterion = nn.BCELoss(reduction="sum")
    pos_weight = torch.tensor([1/2]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="sum")

    if mode == "hyp":
        for hidden_dim in [16,32,64]:
            for lr in [0.0001, 0.0005, 0.001]:

                print("** PARAMETERS **", flush=True)
                print("Hidden:", hidden_dim, flush=True)
                print("LR:", lr, flush=True)
                print("pos_emb:", pos_emb, flush=True)
                
                if pos_emb:
                    model = LSTMClassifier(vocab_size, embedding, hidden_dim, output_dim, n_layers, dropout, pos=True, pos_emb=pos_embedding)
                else:
                    model = LSTMClassifier(vocab_size, embedding, hidden_dim, output_dim, n_layers, dropout, pos=False, pos_emb=pos_embedding)
                model = model.to(device)
                print(model)

                optimizer = optim.Adam(model.parameters(), lr=lr)

                for epoch in range(100):

                    patience = 5
                    min_val_loss = 0.0
                    epoch_loss = 0.0
                    epoch_val_loss = 0.0

                    ## Train loss
                    model.train()
                    for i, batch in enumerate(train_loader, start=1):
                        tokens = batch['tokens'].to(device)
                        pos_tokens = batch['pos_tokens'].to(device)
                        labels = batch['label'].to(device)

                        optimizer.zero_grad()
                        outputs = model(tokens, pos_tokens).to(device)

                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()

                        #if i % 10 == 0:
                        #    print(f"Epoch [{epoch+1}], Step [{i}], Loss: {loss.item():.4f}", flush=True)

                    ## Val loss
                    model.eval()
                    with torch.no_grad():
                        for batch in val_loader:
                            tokens = batch['tokens'].to(device)
                            pos_tokens = batch['pos_tokens'].to(device)
                            labels = batch['label'].to(device)

                            outputs = model(tokens, pos_tokens).to(device)
                            val_loss = criterion(outputs, labels)

                            epoch_val_loss += val_loss.item()

                    print(f'Epoch {epoch+1}, Loss: {epoch_loss}, Val Loss: {epoch_val_loss}', flush=True)
                    
                    if epoch_val_loss < min_val_loss:
                        min_val_loss = epoch_val_loss
                        patience = 5
                    else:
                        if epoch > 10:
                            patience -= 1
                            if patience == 0:
                                break
    else:
        pass
        """model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch['tokens']
                pos_tokens = batch['pos_tokens']
                labels = batch['label']

                outputs = model(tokens)

                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        print(f'Test Accuracy: {correct / total * 100:.2f}%')"""

def main():
    pos = sys.argv[1]
    mode = sys.argv[2]
    if pos == "True":
        pos_emb = True
    else:
        pos_emb = False
    train(pos_emb=pos_emb, mode=mode)

if __name__ == "__main__":
    main()