from nltk.corpus import movie_reviews, stopwords
from nltk import FreqDist
import string
import numpy as np
import pickle
from enum import Enum
import random


class DataLoader:

    def __init__(self):
        self.vocab = []
        self.labelled_vectors = []

    def save_words(self):
        words = [w for w in movie_reviews.words() if
                 w not in string.punctuation and w not in stopwords.words('english')]
        with open('data/words.txt', 'w') as infile:
            infile.write('\n'.join(words))

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
        fileids = movie_reviews.fileids()
        random.shuffle(fileids)
        self.save_labelled_vectors_type(DataType.TRAIN, fileids)
        self.save_labelled_vectors_type(DataType.TEST, fileids)

    def save_labelled_vectors_type(self, type, fileids):
        length = len(fileids)
        first_index = 0 if type == DataType.TRAIN else int(0.6 * length) + 1
        last_index = int(0.6 * length) if type == DataType.TRAIN else length
        count = 0
        count_pos = 0
        count_neg = 0
        reviews = []
        for i in range(first_index, last_index, 1):
            fileid = fileids[i]
            label = 1 if movie_reviews.categories(fileid)[0] == 'pos' else 0
            reviews.append((self.to_vector(movie_reviews.words(fileids=fileid)), label))
            count += 1
            if label == 1:
                count_pos += 1
            else:
                count_neg += 1
            if count % 10 == 0:
                print('{}/{}. Label: {}'.format(count, last_index - first_index, label))

        with open('data/labelled_vectors_{}'.format(type), 'wb+') as infile:
            pickle.dump(reviews, infile)

        print("Pos: {}; Neg: {}".format(count_pos, count_neg))

    def load_labelled_vectors(self, type):
        with open('data/labelled_vectors_{}'.format(type), 'rb') as infile:
            return pickle.load(infile)

    def mini_batches(self, size):
        data = self.load_labelled_vectors(DataType.TRAIN)
        for i in range(int(len(data)/size)):
            yield data[i*size:i*size+size]

    def display_labelled_vectors(self, type):
        for r in self.load_labelled_vectors(type):
            print('{}: {}'.format(self.from_vector(r[0]), r[1]))

    def load_reviews(self):
        return self.load_labelled_vectors(DataType.TRAIN), self.load_labelled_vectors(DataType.TEST)


class DataType(Enum):
    TEST = 1
    TRAIN = 2


if __name__ == '__main__':
    # acc: 530 ca.
    dl = DataLoader()
    # dl.save_vocab(500)
    # dl.save_labelled_vectors()
    dl.load_vocab()
    dl.display_labelled_vectors(DataType.TRAIN)
