import os
import argparse

import iob
from poldeepner import PolDeepNer
from pretrained import load_pretrained_model
from utils import NestedReport
from seqeval.metrics import f1_score

parser = argparse.ArgumentParser(description='Evaluate given model against annotated document in IOB format.')
parser.add_argument('-m', required=True, metavar='name', help='model name', default='n82')
args = parser.parse_args()


root = os.path.dirname(os.path.abspath(__file__))
path_data = os.path.join(root, "..", "data")
path_eval = os.path.join(path_data, "kpwr-n82-test-sample.iob")

try:
    model = load_pretrained_model(args.m)
    ner = PolDeepNer(model)

    label_true, label_pred = [], []
    x_test, y_test = iob.load_data_and_labels(path_eval)
    for x, y in zip(x_test, y_test):
        pred = ner.process_sentence(x)
        label_true.append(y)
        label_pred.append(pred)

    report = NestedReport(label_true, label_pred)
    print(str(report))

    #score = f1_score(label_true, label_pred)
    #print(score)

except Exception as e:
    print("[ERROR] %s" % str(e))
