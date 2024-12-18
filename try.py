import spacy
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

# Load the Danish spaCy model
nlp = spacy.load("da_core_news_sm")

df = pd.read_csv('cleaned_final.csv')

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
    unique_pos_tags = extract_unique_pos_tags(row['Rewritten Text'], unique_pos_tags)
    unique_pos_tags = extract_unique_pos_tags(row['Combined Text'], unique_pos_tags)

# Convert the set to a sorted list
unique_pos_tags_list = sorted(list(unique_pos_tags))

# Print the result
print("Unique POS Tags:", unique_pos_tags_list)

# Define standard POS tag list
standard_pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 
                     'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SPACE', 'SYM', 'VERB', 'X']

# Create a dictionary with indices for standard POS tags
pos_tag_dict = {tag: idx for idx, tag in enumerate(standard_pos_tags)}

# One-Hot Encode POS tags
one_hot_encoder = OneHotEncoder(categories=[standard_pos_tags], sparse_output=False)
encoded_pos_tags = one_hot_encoder.fit_transform(np.array(unique_pos_tags_list).reshape(-1, 1))

# Map POS tags to corresponding row indices in the embedding table
pos_embedding_mapping = {tag: int(np.argmax(encoded_pos_tags[idx])) for idx, tag in enumerate(unique_pos_tags_list)}

# Print the results
print("Unique POS Tags:", unique_pos_tags_list)
print("POS Tag Dictionary:", pos_tag_dict)
print("POS Embedding Mapping:", pos_embedding_mapping)