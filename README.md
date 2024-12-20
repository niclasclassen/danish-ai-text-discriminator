# Can Part-of-Speech Embeddings Help Us Identify Generative AI Text?

Project in the course Advanced Natural Language Processing @ IT University of Copenhagen.

We use real posts on Reddit and a generative AI model (LLaMA) to generate human and AI-generated texts. With part-of-speech tagging and a simple LSTM architecture, we attempt at creating a generative AI text discriminator. For the main experimental code, see the branch "hyp-tuning".

## To run the code

Run the */models/training.py* script followed by two command arguments:
- 1: True/False (Whether to use POS model or baseline, True for POS model)
- 2: "hyp" for hyperparameter tuning, else it will do the final training using config files.

E.g.: 
For final training of POS model:
*python training.py True train*
