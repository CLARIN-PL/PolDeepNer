# -*- coding: utf-8 -*-
"""
Preprocessors.
"""
from __future__ import absolute_import

import hashlib
import numpy as np
import os
import re
import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from .utils import Vocabulary


def normalize_number(text):
    return re.sub(r'[0-9０１２３４５６７８９]', r'0', text)


class VectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, embedding, use_char=False, lower=False):
        self._embedding = embedding
        self._label_vocab = Vocabulary(lower=lower, unk_token=False)
        self._char_vocab = Vocabulary(lower=False)
        self._use_char = use_char
        self._embedding_md5 = self._embedding.md5

    def fit(self, sentences, labels):
        self._label_vocab.add_documents(labels)

        if self._use_char:
            for sentence in sentences:
                self._char_vocab.add_documents(sentence)
        self._label_vocab.build()
        self._char_vocab.build()
        return self

    def get_word_vector(self, word):
        return self._embedding.get_numpy_vector(word)

    def transform(self, sentences, labels=None):
        vector_vocab = [[self.get_word_vector(word) for word in sentence] for sentence in sentences]
        vector_vocab = pad_sequences(vector_vocab, dtype='float32', padding='post')

        if self._use_char:
            char_ids = [[self._char_vocab.doc2id(word) for word in sentence] for sentence in sentences]
            char_ids = pad_nested_sequences(char_ids, dtype='float32')
            features = [vector_vocab, char_ids]
        else:
            features = vector_vocab

        if labels is not None:
            y = [self._label_vocab.doc2id(doc) for doc in labels]
            y = pad_sequences(y, padding='post')
            y = to_categorical(y, self.label_size).astype(int)
            # In 2018/06/01, to_categorical is a bit strange.
            # >>> to_categorical([[1,3]], num_classes=4).shape
            # (1, 2, 4)
            # >>> to_categorical([[1]], num_classes=4).shape
            # (1, 4)
            # So, I expand dimensions when len(y.shape) == 2.
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

    @property
    def embedding_name(self):
        return self._embedding.name

    @property
    def word_vector_len(self):
        length = self.embedding_vector_len
        return length

    @property
    def embedding_vector_len(self):
        return len(self._embedding)

    @property
    def vector_len(self):
        return self.word_vector_len

    @property
    def labels(self):
        return self._label_vocab.vocab

    @property
    def label_size(self):
        return len(self._label_vocab)

    @property
    def char_vocab_size(self):
        return len(self._char_vocab)

    def save(self, file_path):
        ''' Embedding object has to be removed from preprocessor in order to dump it into a file. Joblib can't handle
        dumping Fasttext/Word2Vec objects '''
        self._embedding = None
        joblib.dump(self, file_path)

    @staticmethod
    def load(preprocessor_file_path, embedding):
        p = joblib.load(preprocessor_file_path)
        if embedding.md5 == p._embedding_md5:
            p._embedding = embedding
            return p
        else:
            raise ValueError("FastText/Word2Vec embedding provided for load is different than one used by preprocessor. "
                             "Preprocessor embedding name: ", p._emb_name)


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
