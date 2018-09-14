"""
Model definition.
"""
import json

from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, TimeDistributed, GRU
from keras.layers.merge import Concatenate
from keras.models import Model, model_from_json
from keras_contrib.layers import CRF


def save_model(model, weights_file, params_file):
    with open(params_file, 'w') as f:
        params = model.to_json()
        json.dump(json.loads(params), f, sort_keys=True, indent=4)
        model.save_weights(weights_file)


def load_model(weights_file, params_file):
    with open(params_file) as f:
        model = model_from_json(f.read(), custom_objects={'CRF': CRF})
        model.load_weights(weights_file)

    return model


class BiLSTMCRF(object):
    """A Keras implementation of BiLSTM-CRF for sequence labeling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """

    def __init__(self,
                 num_labels,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 word_lstm_size=100,
                 char_lstm_size=25,
                 fc_dim=100,
                 dropout=0.5,
                 embeddings=None,
                 use_char=False,
                 use_crf=True,
                 use_fasttext=False,
                 nn_type="GRU",
                 input_size=300):
        """Build a Bi-LSTM CRF model.

        Args:
            word_vocab_size (int): word vocabulary size.
            char_vocab_size (int): character vocabulary size.
            num_labels (int): number of entity labels.
            word_embedding_dim (int): word embedding dimensions.
            char_embedding_dim (int): character embedding dimensions.
            word_lstm_size (int): character LSTM feature extractor output dimensions.
            char_lstm_size (int): word tagger LSTM output dimensions.
            fc_dim (int): output fully-connected layer size.
            dropout (float): dropout rate.
            embeddings (numpy array): word embedding matrix.
            use_char (boolean): add char feature.
            use_crf (boolean): use crf as last layer.
            nn_type (String): NN type: GRU or LSTM.
            input_size (int): input size of the first layer.
        """
        super(BiLSTMCRF).__init__()
        self._char_embedding_dim = char_embedding_dim
        self._word_embedding_dim = word_embedding_dim
        self._char_lstm_size = char_lstm_size
        self._word_lstm_size = word_lstm_size
        self._fc_dim = fc_dim
        self._dropout = dropout
        self._use_char = use_char
        self._use_crf = use_crf
        self._embeddings = embeddings
        self._num_labels = num_labels
        self._use_fasttext = use_fasttext
        self._nn_type = nn_type
        self._input_size = input_size

    def build(self):
        # build word embedding
        words = Input(batch_shape=(None, None, self._input_size), dtype='float32', name='word_input')
        inputs = [words]

        # build character based word embedding
        if self._use_char:
            char_ids = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
            inputs.append(char_ids)
            char_embeddings = Embedding(input_dim=self._char_vocab_size,
                                        output_dim=self._char_embedding_dim,
                                        mask_zero=True,
                                        name='char_embedding')(char_ids)
            char_embeddings = TimeDistributed(Bidirectional(GRU(self._char_lstm_size)))(char_embeddings)
            word_embeddings = Concatenate()([words, char_embeddings])
        else:
            word_embeddings = words
        word_embeddings = Dropout(self._dropout)(word_embeddings)

        if self._nn_type == "GRU":
            z = Bidirectional(GRU(units=self._word_lstm_size, return_sequences=True))(word_embeddings)
        elif self._nn_type == "LSTM":
            z = Bidirectional(LSTM(units=self._word_lstm_size, return_sequences=True))(word_embeddings)
        else:
            raise Exception("Unknown NN type: %s (expected GRU or LSTM)" % self._nn_type)

        z = Dense(self._fc_dim, activation='tanh')(z)

        if self._use_crf:
            crf = CRF(self._num_labels, sparse_target=False)
            loss = crf.loss_function
            pred = crf(z)
        else:
            loss = 'categorical_crossentropy'
            pred = Dense(self._num_labels, activation='softmax')(z)

        model = Model(inputs=inputs, outputs=pred)

        print(model.summary())

        return model, loss
