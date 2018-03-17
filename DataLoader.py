from nltk.corpus import movie_reviews, stopwords
from nltk import FreqDist
import string
import numpy as np
import random
import pickle


class DataLoader:

    def __init__(self):
        self.vocab = []
        self.labelled_vectors = []

    @staticmethod
    def save_words(self):
        words = [w for w in movie_reviews.words() if
                 w not in string.punctuation and w not in stopwords.words('english')]
        with open('data/words.txt', 'w') as infile:
            infile.write('\n'.join(words))

    @staticmethod
    def save_vocab(self, size):
        with open('data/words.txt', 'r') as word_file:
            words = word_file.read().splitlines()
            freq_dist = FreqDist(words)
            words = [w for w, c in freq_dist.most_common(size)]
        with open('data/vocab.txt', 'w') as vocab_file:
            vocab_file.write('\n'.join(words))

    def load_vocab(self):
        with open('data/vocab.txt', 'r') as infile:
            self.vocab = infile.read().splitlines()

    def to_vector(self, words):
        freq_dist = FreqDist(words)
        v = [freq_dist[w] if w in words else 0 for w in self.vocab]
        return np.array(v).reshape(len(self.vocab), 1)

    def from_vector(self, v):
        return " ".join([self.vocab[i] for i in range(len(v)) if v[i] > 0])

    def save_labelled_vectors(self):
        self.load_vocab()
        last_train_index = int(0.6 * len(movie_reviews.fileids()))
        posids = movie_reviews.fileids('pos')[:last_train_index]
        negids = movie_reviews.fileids('neg')[:last_train_index]
        length = len(posids) + len(negids)
        count = 0
        reviews = []
        for i in range(len(posids)):
            reviews.append((self.to_vector(movie_reviews.words(fileids=[posids[i]])), 1))
            count += 1
            if count % 10 == 0:
                print('{}/{}'.format(count, length))

        for i in range(len(negids)):
            reviews.append((self.to_vector(movie_reviews.words(fileids=[negids[i]])), 0))
            count += 1
            if count % 10 == 0:
                print('{}/{}'.format(count, length))

        with open('data/labelled_vectors', 'wb') as infile:
            pickle.dump(reviews, infile)

    def load_labelled_vectors(self):
        with open('data/labelled_vectors', 'rb') as infile:
            self.labelled_vectors = pickle.load(infile)

    def display_labelled_vectors(self):
        for r in self.labelled_vectors:
            print('{}: {}'.format(self.from_vector(r[0]), r[1]))

    def load_reviews(self):
        # TODO
        pass

if __name__ == '__main__':
    dl = DataLoader()
    dl.load_vocab()
    dl.load_labelled_vectors()
    dl.display_labelled_vectors()
