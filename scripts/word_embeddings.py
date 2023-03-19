import nltk
from gensim.models import KeyedVectors

def extract_subset(words = ['king', 'queen', 'oil', 'gas', 'happy', 'sad', 'city', 'town', 'village', 'country', 'continent', 'petroleum', 'joyful']):
    embeddings = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary = True)
    f = open('capitals.txt', 'r').read()
    set_words = set(nltk.word_tokenize(f))
    for w in words:
        set_words.add(w)

    word_embeddings = get_word_embeddings(embeddings)
    print(len(word_embeddings))
    pickle.dump( word_embeddings, open( "word_embeddings_subset.p", "wb" ) )


def get_word_embeddings(embeddings, set_words):
    word_embeddings = {}
    for word in embeddings.vocab:
        if word in set_words:
            word_embeddings[word] = embeddings[word]
    return word_embeddings
