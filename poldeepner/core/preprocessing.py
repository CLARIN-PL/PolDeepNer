# -*- coding: utf-8 -*-
"""
Preprocessors.

Based on https://github.com/Hironsan/anago
"""
from numpy.ma import concatenate
from pyfasttext import FastText
from allennlp.commands.elmo import ElmoEmbedder

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib

from utils import Vocabulary


elmo_catched = {}


def get_elmo(path):
    if path in elmo_catched:
        print(">>> ELMo found in the cache")
        return elmo_catched[path]
    else:
        print(">>> ELMo was cached")
        options_file = path + "/options.json"
        weight_file = path + "/weights.hdf5"
        elmo = ElmoEmbedder(options_file, weight_file, 1)
        elmo_catched[path] = elmo
        return elmo


class FastTextEmbeddings:

    def __init__(self, path):
        self.fasttext = FastText(path)

    def generate(self, sentence):
        return [self.fasttext.get_numpy_vector(word) for word in sentence]

    def size(self):
        return 300


class ElmoEmbeddings:

    def __init__(self, path):
        self.elmo = get_elmo(path)

    def generate(self, sentence):
        return self.elmo.embed_sentence(sentence)[2]

    def size(self):
        return 1024


class ElmoAverageEmbeddings:

    def __init__(self, path):
        self.elmo = get_elmo(path)

    def generate(self, sentence):
        return np.mean(self.elmo.embed_sentence(sentence), axis=0)

    def size(self):
        return 1024


class ElmoConcatEmbeddings:

    def __init__(self, path):
        self.elmo = get_elmo(path)

    def generate(self, sentence):
        vs = self.elmo.embed_sentence(sentence)
        return [concatenate(v) for v in zip(vs[0], vs[1], vs[2])]

    def size(self):
        return 1024 * 3


class VectorTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, embeddings, use_char=True, lower=False):
        self._embeddings = self.create_language_model(embeddings)
        self._label_vocab = Vocabulary(lower=lower, unk_token=False)
        self._char_vocab = Vocabulary(lower=False)
        self._use_char = use_char

    def size(self):
        return len(self._embeddings.generate(["."])[0])

    def create_language_model(self, description):
        parts = description.split(":")
        if parts[0] == "ft":
            return FastTextEmbeddings(parts[1])
        elif parts[0] == "elmo":
            return ElmoEmbeddings(parts[1])
        elif parts[0] == "elmo-avg":
            return ElmoAverageEmbeddings(parts[1])
        elif parts[0] == "elmo-concat":
            return ElmoConcatEmbeddings(parts[1])
        else:
            raise Exception("Unknown type of language model %s" % parts[0])

    def fit(self, sentences, labels):
        self._label_vocab.add_documents(labels)
        if self._use_char:
            for sentence in sentences:
                self._char_vocab.add_documents(sentence)
        self._label_vocab.build()
        self._char_vocab.build()
        return self

    def transform(self, sentences, labels=None):

        vector_vocab = [self._embeddings.generate(sentence) for sentence in sentences]
        vector_vocab = pad_sequences(vector_vocab, dtype='float32', padding='post')

        if self._use_char:
            char_ids = [[self._char_vocab.doc2id(w) for w in doc] for doc in X]
            char_ids = pad_nested_sequences(char_ids)
            features = [vector_vocab, char_ids]
        else:
            features = vector_vocab

        if labels is not None:
            y = [self._label_vocab.doc2id(doc) for doc in labels]
            y = pad_sequences(y, padding='post')
            y = to_categorical(y, self.label_size).astype(int)
            y = y if len(y.shape) == 3 else np.expand_dims(y, axis=0)
            return features, y
        return features

    def inverse_transform(self, y, lengths=None):
        """Return label strings.

        Args:
            y: label id matrix.
            lengths: sentences length.

        Returns:
            list: list of list of strings.
        """
        y = np.argmax(y, -1)
        inverse_y = [self._label_vocab.id2doc(ids) for ids in y]
        if lengths is not None:
            inverse_y = [iy[:l] for iy, l in zip(inverse_y, lengths)]

        return inverse_y

    def load_embeddings(self, embeddings):
        self._embeddings = self.create_language_model(embeddings)

    @property
    def label_size(self):
        return len(self._label_vocab)

    def save(self, file_path):
        emb = self._embeddings
        self._embeddings = None
        joblib.dump(self, file_path)
        self._embeddings = emb

    @classmethod
    def load(cls, file_path, embeddings):
        p = joblib.load(file_path)
        p.load_embeddings(embeddings)
        return p


def pad_nested_sequences(sequences, dtype='int32'):
    """Pads nested sequences to the same length.

    This function transforms a list of list sequences
    into a 3D Numpy array of shape `(num_samples, max_sent_len, max_word_len)`.

    Args:
        sequences: List of lists of lists.
        dtype: Type of the output sequences.

    # Returns
        x: Numpy array.
    """
    max_sent_len = 0
    max_word_len = 0
    for sent in sequences:
        max_sent_len = max(len(sent), max_sent_len)
        for word in sent:
            max_word_len = max(len(word), max_word_len)

    x = np.zeros((len(sequences), max_sent_len, max_word_len)).astype(dtype)
    for i, sent in enumerate(sequences):
        for j, word in enumerate(sent):
            x[i, j, :len(word)] = word

    return x
