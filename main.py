import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import compute_pca, get_vectors

def main():
    parser = argparse.ArgumentParser(description='Compute PCA for Word Embeddings')

    parser.add_argument('--no_components',
                       default=2,
                       type=int,
                       help='Number of PCA Components')

    parser.add_argument('--words', type=lambda s: [item.strip() for item in s.split(',')], help='Selected words')

    parser.add_argument('--data_path',
                       default='./data/word_embeddings_subset.p',
                       help='Word Embeddings Data Path')

    args = parser.parse_args()

    assert(len(args.words) >= args.no_components)

    word_embeddings = pickle.load(open(args.data_path, "rb"))

    X = get_vectors(word_embeddings, args.words)

    result = compute_pca(X, args.no_components)
    print(f"New Shape after PCA: {result.shape}")

    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(args.words):
        plt.annotate(word, xy=(result[i, 0] - 0.05, result[i, 1] + 0.1))

    plt.show()

if __name__ == "__main__":
    main()
