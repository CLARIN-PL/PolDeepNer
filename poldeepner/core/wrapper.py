"""
Wrapper class.

Based on https://github.com/Hironsan/anago
"""
from seqeval.metrics import f1_score

from models import BiLSTMCRF, save_model, load_model
from preprocessing import VectorTransformer
from trainer import Trainer

import os


class Sequence(object):

    def __init__(self,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 word_lstm_size=100,
                 char_lstm_size=25,
                 fc_dim=100,
                 dropout=0.5,
                 embeddings=None,
                 use_char=True,
                 use_crf=True,
                 initial_vocab=None,
                 lower=False,
                 optimizer='adam',
                 nn_type='GRU',
                 input_size=300):

        self.model = None
        self.p = None
        self.tagger = None

        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.word_lstm_size = word_lstm_size
        self.char_lstm_size = char_lstm_size
        self.fc_dim = fc_dim
        self.dropout = dropout
        self.embeddings = embeddings
        self.use_char = use_char
        self.use_crf = use_crf
        self.initial_vocab = initial_vocab
        self.optimizer = optimizer
        self.lower = lower
        self.nn_type = nn_type
        self.input_size = input_size

    def fit(self, x_train, y_train, fasttext_path, x_valid=None, y_valid=None,
            epochs=1, batch_size=32, verbose=1, callbacks=None, shuffle=True):
        """Fit the model for a fixed number of epochs.

        Args:
            x_train: list of training model.
            y_train: list of training target (label) model.
            x_valid: list of validation model.
            y_valid: list of validation target (label) model.
            batch_size: Integer.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
            epochs: Integer. Number of epochs to train the model.
            verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
            shuffle: Boolean (whether to shuffle the training model
                before each epoch). `shuffle` will default to True.
        """

        p = VectorTransformer(fasttext_path, use_char=self.use_char)
        p.fit(x_train, y_train)

        model = BiLSTMCRF(num_labels=p.label_size,
                          word_lstm_size=self.word_lstm_size,
                          char_lstm_size=self.char_lstm_size,
                          fc_dim=self.fc_dim,
                          dropout=self.dropout,
                          use_char=self.use_char,
                          use_crf=self.use_crf,
                          nn_type=self.nn_type,
                          input_size=self.input_size)
        model, loss = model.build()

        model.compile(loss=loss, optimizer=self.optimizer)

        trainer = Trainer(model, preprocessor=p)
        trainer.train(x_train, y_train, x_valid, y_valid,
                      epochs=epochs, batch_size=batch_size,
                      verbose=verbose, callbacks=callbacks,
                      shuffle=shuffle)

        self.p = p
        self.model = trainer.best_model
        print("Best model report\n")
        print(trainer.best_model_report)

    def score(self, x_test, y_test):
        """Returns the f1-micro score on the given test model and labels.

        Args:
            x_test : array-like, shape = (n_samples, sent_length)
            Test samples.

            y_test : array-like, shape = (n_samples, sent_length)
            True labels for x.

        Returns:
            score : float, f1-micro score.
        """
        if self.model:
            x_test = self.p.transform(x_test)
            lengths = map(len, y_test)
            y_pred = self.model.predict(x_test)
            y_pred = self.p.inverse_transform(y_pred, lengths)
            score = f1_score(y_test, y_pred)
            return score
        else:
            raise OSError('Could not find a model. Call load(dir_path).')

    def predict(x_test):
        if self.model:
            x_test = self.p.transform(x_test)
            y_pred = self.model.predict(x_test)
            y_pred = self.p.inverse_transform(y_pred, lengths)
            print(x_test)
            print(y_pred)


    def predict_to_iob(self, input_path, output_path):
        input_file = open(input_path, 'r')

        output_file = open(output_path, 'w')
        sentence, iob_lines, true_labels = [], [], []

        for input_line in input_file:
            if "DOCSTART CONFIG" in input_line or "DOCSTART FILE" in input_line:
                continue
            if input_line == '\n':
                x_test = self.p.transform([sentence])
                lengths = map(len, [true_labels])
                y_pred = self.model.predict(x_test)
                y_pred = self.p.inverse_transform(y_pred, lengths)
                for prediction, iob_line in zip(y_pred[0], iob_lines):
                    output_line = ''
                    for pos in iob_line:
                        output_line += pos + '\t'
                    output_line += prediction
                    output_file.write(output_line + '\n')
                output_file.write('\n')
                sentence = []
                iob_lines = []
            else:
                iob_line = input_line.split(sep='\t')[0:3]
                iob_lines.append(iob_line)
                sentence.append(input_line.split(sep='\t')[0])
                true_labels.append(input_line.split(sep='\t')[3])

    def predict_sentence(self, sentence):
        x_test = self.p.transform([sentence])
        lengths = [len(sentence)]
        y_pred = self.model.predict(x_test)
        y_pred = self.p.inverse_transform(y_pred, lengths)
        #print(y_pred)
        return y_pred[0]

    def save(self, weights_file, params_file, preprocessor_file):
        self.p.save(preprocessor_file)
        save_model(self.model, weights_file, params_file)

    @classmethod
    def load(cls, model_path, fasttext):
        self = cls()
        weights_file = os.path.join(model_path, "weights.pkl")
        params_file = os.path.join(model_path, "params.pkl")
        preprocessor_file = os.path.join(model_path, "preprocessor.pkl")
        self.p = VectorTransformer.load(preprocessor_file, fasttext)
        self.model = load_model(weights_file, params_file)

        return self
