"""
Custom callbacks.
"""
from keras.callbacks import Callback
from seqeval.metrics import f1_score

from utils import get_lengths, NestedReport


class F1score(Callback):

    def __init__(self, seq, preprocessor=None):
        super(F1score, self).__init__()
        self.seq = seq
        self.p = preprocessor
        self.best_model = None
        self._best_score = 0.0
        self._best_report = ""

    def on_epoch_end(self, epoch, logs={}):
        label_true = []
        label_pred = []
        for i in range(len(self.seq)):
            x_true, y_true = self.seq[i]
            lengths = get_lengths(y_true)
            y_pred = self.model.predict_on_batch(x_true)

            y_true = self.p.inverse_transform(y_true, lengths)
            y_pred = self.p.inverse_transform(y_pred, lengths)

            label_true.extend(y_true)
            label_pred.extend(y_pred)
        score = f1_score(label_true, label_pred)
        print(' - f1: {:04.2f}'.format(score * 100))
        print(NestedReport(label_true, label_pred))
        logs['f1'] = score
        if self._best_score < score:
            self.best_model = self.model
            self._best_score = score
            self._best_report = report

    def get_best_model(self):
        return self.best_model

    def get_best_model_report(self):
        return self._best_report
