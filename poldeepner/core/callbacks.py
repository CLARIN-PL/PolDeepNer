"""
Custom callbacks.
"""
import os

from keras.callbacks import Callback
from models import save_model

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

        report = NestedReport(label_true, label_pred)
        print(report)
        f1_score = report.f1_score
        logs['f1'] = f1_score
        if self._best_score < f1_score:
            self.best_model = self.model
            self._best_score = f1_score
            self._best_report = report

    def get_best_model(self):
        return self.best_model

    def get_best_model_report(self):
        return self._best_report


class ModelEpochSaver(Callback):

    def __init__(self, path, preprocessor):
        self.path = path
        self.preprocessor = preprocessor
        self.epoch = 1

    def on_epoch_end(self, epoch, logs=None):
        epoch_str = "%03d" % self.epoch
        folder = self.path + epoch_str

        if not os.path.exists(folder):
            os.mkdir(folder)

        model_weights = os.path.join(folder, "weights.pkl")
        model_params = os.path.join(folder, "params.pkl")
        model_preprocessor = os.path.join(folder, "preprocessor.pkl")

        self.preprocessor.save(model_preprocessor)
        save_model(self.model, model_weights, model_params)

        self.epoch += 1
