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
import spacy
from sklearn.preprocessing import OneHotEncoder

def get_embs():
    # Pretrained word embeddings
    w2v_path = "../embeddings/da/w2v.bin"
    w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True, unicode_errors='ignore')
    unk_vector = np.random.uniform(-0.01, 0.01, w2v.vector_size)
    unk_id = w2v.add_vector('<UNK>', unk_vector)
    padding_vector = np.random.uniform(-0.01, 0.01, w2v.vector_size)
    pad_id = w2v.add_vector('<PAD>', padding_vector)
    embedding = nn.Embedding.from_pretrained(torch.FloatTensor(w2v.vectors))
    vocab = {word: idx for idx, word in enumerate(w2v.index_to_key)}
    return embedding, vocab, unk_id, pad_id

def get_pos_embs(df):
    # Load the Danish spaCy model
    nlp = spacy.load("da_core_news_sm")

    # Function to extract unique POS tags from text
    def extract_unique_pos_tags(text, unique_pos_tags):
        if pd.isna(text):
            return unique_pos_tags
        doc = nlp(text)
        for token in doc:
            unique_pos_tags.add(token.pos_)
        return unique_pos_tags

    # Initialize a set for unique POS tags
    unique_pos_tags = set()

    # Iterate through rows and extract POS tags from both columns
    for _, row in df.iterrows():
        unique_pos_tags = extract_unique_pos_tags(row['Text'], unique_pos_tags)

    # Convert the set to a sorted list
    unique_pos_tags_list = sorted(list(unique_pos_tags))

    # Print the result
    #print("Unique POS Tags:", unique_pos_tags_list)

    # Define standard POS tag list
    standard_pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 
                        'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SPACE', 'SYM', 'VERB', 'X', 'PAD']

    # Create a dictionary with indices for standard POS tags
    pos_tag_dict = {tag: idx for idx, tag in enumerate(standard_pos_tags)}

    # One-Hot Encode POS tags
    one_hot_encoder = OneHotEncoder(categories=[standard_pos_tags], sparse_output=False)
    encoded_pos_tags = one_hot_encoder.fit_transform(np.array(unique_pos_tags_list).reshape(-1, 1))

    print(encoded_pos_tags)
    embedding = nn.Embedding.from_pretrained(torch.FloatTensor(encoded_pos_tags))

    # Map POS tags to corresponding row indices in the embedding table
    pos_embedding_mapping = {tag: int(np.argmax(encoded_pos_tags[idx])) for idx, tag in enumerate(unique_pos_tags_list)}
    
    return embedding, pos_embedding_mapping, len(standard_pos_tags)-2, len(standard_pos_tags)-1