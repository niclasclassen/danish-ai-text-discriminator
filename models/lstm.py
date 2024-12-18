import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from itertools import chain
from gensim.models import KeyedVectors

w2v_path = "../embeddings/da/w2v.bin"

### Very basic LSTM model for text classification
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        tokens = self.tokenizer(text)
        padded_tokens = tokens + [0] * (self.max_len - len(tokens)) if len(tokens) < self.max_len else tokens[:self.max_len]

        return {
            'input': torch.tensor(padded_tokens, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float),
        }
    
#data = pd.read_csv("../reddit.csv") # Load the data
data = pd.read_csv("../rephrased_texts.csv")
human_df = data[["Original Title", "Original Text"]]
human_df["label"] = 0
ai_df = data[["Rewritten Text"]]
ai_df["label"] = 0

# add a column "generated" to the dataframe (0 for real, 1 for generated) this we will probably just modify directly in the csv
# so it can be removed when we do the actual training
data['generated'] = 0

# Tokenize text
def simple_tokenizer(text):
    return text.lower().split()

# Pretrained word embeddings
w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True, unicode_errors='ignore')
unk_vector = np.random.uniform(-0.01, 0.01, w2v.vector_size)
unk_id = w2v.add_vector('<UNK>', unk_vector)
embedding = nn.Embedding.from_pretrained(w2v.vectors)
vocab = {word: idx for idx, word in enumerate(w2v.index_to_key)}

# Tokenize and build vocabulary
#all_tokens = list(chain.from_iterable(simple_tokenizer(text) for text in data['text']))
#vocab = {word: idx+1 for idx, (word, _) in enumerate(Counter(all_tokens).most_common())}

# Map words to integers
def tokenizer(text):
    return [vocab.get(word, unk_id) for word in simple_tokenizer(text)]

###############
# Train model #
###############

max_len = 100  # Adjust based on your dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(data['text'], data['generated'], test_size=0.2, random_state=42)

train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, max_len)
test_dataset = TextDataset(test_texts.tolist(), test_labels.tolist(), tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = hidden[-1]  # Take the output of the last LSTM layer
        out = self.fc(hidden)
        return self.sigmoid(out)

vocab_size = len(vocab) + 1  # Add 1 for padding index
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
