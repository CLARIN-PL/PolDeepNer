from gensim.models import KeyedVectors
import hashlib
import os
from pyfasttext import FastText



def load_embedding(embedding_path):
    """
    :param embedding_path: path to FastText or KeyedVectors embedding
    :return: embedding object
    """
    if embedding_path.endswith('.bin'):
        embedding = FastTextWrapper(embedding_path)
    elif embedding_path.endswith('.vec'):
        embedding = KeyedVectorsWrapper(embedding_path)
    else:
        raise UnsupportedEmbeddingFormat(embedding_path)

    return embedding


class EmbeddingWrapper(object):
    """
    Abstract class enclosing embeddings
    """
    def __init__(self, path):
        self._emb_name = os.path.basename(path)
        self._md5 = self._hash_emb(path)

    def __len__(self):
        raise NotImplemented

    def get_numpy_vector(self, word):
        raise NotImplemented

    @property
    def name(self):
        return self._emb_name

    @property
    def md5(self):
        return self._md5

    @staticmethod
    def _hash_emb(embedding_path):
        md5 = hashlib.md5()
        with open(embedding_path, 'rb') as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                md5.update(data)
        return md5.digest()


class KeyedVectorsWrapper(EmbeddingWrapper):
    """
    Contains the KeyedVectors object, name of the file from which the embeddings were loaded as well as its md5.
    """
    def __init__(self, word2vec_path):
        super(KeyedVectorsWrapper, self).__init__(word2vec_path)
        self._word2vec = KeyedVectors.load_word2vec_format(word2vec_path)
        self._word_output_len = self._word2vec["unk"].shape[0]

    def __len__(self):
        return self._word_output_len

    def get_numpy_vector(self, word):
        try:
            return self._word2vec[word]
        except KeyError:
            return self._word2vec['unk']

    def emb_obj(self):
        """Returns embedding dict"""
        return self._word2vec


class FastTextWrapper(EmbeddingWrapper):
    """
    Contains the KeyedVectors object, name of the file from which the embeddings were loaded as well as its md5.
    """
    def __init__(self, fasttext_path):
        super(FastTextWrapper, self).__init__(fasttext_path)
        self._fasttext = FastText(fasttext_path)
        self._word_output_len = self.get_numpy_vector("checklen").shape[0]

    def __len__(self):
        return self._word_output_len

    def get_numpy_vector(self, word):
        return self._fasttext.get_numpy_vector(word)

    def emb_obj(self):
        """
        Returns FastText object
        """
        return self._fasttext


class UnsupportedEmbeddingFormat(Exception):
    def __init__(self, message):
        super(UnsupportedEmbeddingFormat, self).__init__(message)
