# Compute PCA for Word Embeddings
This project has an implementation of Principal Components Analysis (PCA) in Python. The goal of this project is to use Principal Components Analysis (PCA) to reduce the dimensions of 300-dimensional space of word embeddings to be able to visualize it.

## Data
You can download the full word embedding dataset `GoogleNews-vectors-negative300.bin.gz` from [Google News Page](https://code.google.com/archive/p/word2vec/).

Or you can load a pickle file which is a subset version (300 dimensions) of the full word embedding dataset extracted under this path `./data/word_embeddings_subset.p`.

## Run
```
python main.py --no_components 2 \
               --words "oil,gas,happy,sad,city,town,village,country,continent,petroleum,joyful" \
               --data_path ./data/word_embeddings_subset.p
```
