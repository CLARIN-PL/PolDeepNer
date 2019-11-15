import iob
from poldeepner import PolDeepNer
from utils import NestedReport
from seqeval.metrics import f1_score

eval_path = "/ha1/czuk/nlp/corpora/pwr/kpwr-workdir/kpwr-1.2.7-names-disamb-n82-flatten-iob/kpwr-ner-n82-test.iob"

ner = PolDeepNer()
label_true, label_pred = [], []
x_test, y_test = iob.load_data_and_labels(eval_path)
for x, y in zip(x_test, y_test):
    pred = ner.process_sentence(x)
    label_true.append(y)
    label_pred.append(pred)

report = NestedReport(label_true, label_pred)
print(str(report))
score = f1_score(label_true, label_pred)
print(score)
