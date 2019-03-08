"""
Utility functions.
"""
import math
from collections import Counter
import os
import subprocess

import numpy as np
from keras.utils import Sequence

from annotation import Annotation

def load_tokenised():
    dir_path = os.getcwd()
    with open('/../data/tmp_tokenised.txt', 'r') as f:
        data = []
        sentence = []
        for line in f:
            line_splitted = line.split('\t')
            if line_splitted[1] == 'newline':
                data.append(sentence)
                sentence = []
                continue
            else:
                sentence.append(line_splitted[0])
    os.remove('./../data/tmp_tokenised.txt')
    return data


def tokenise_file(file_path):
    print("########################################################")
    print(os.path.dirname(file_path))
    print("___________________________________________________________")
    cwd = os.getcwd()
    print(cwd)
    print(os.path.split(cwd)[0])
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(os.path.split(cwd)[0] + '/data/tmp_tokenised.txt')

    p = subprocess.Popen('toki-app -f \$orth\\t\$ws\\n < /mnt/big_one/gawor/data/inwokacja.txt > ' + os.path.split(cwd)[0] + '/data/tmp_tokenised.txt')
    p.wait()
    return load_tokenised()


def wrap_annotations(sentences):
    annotations = []
    tid = 0
    for sid, labels in enumerate(sentences):
        for idx, label in enumerate(labels):
            for ann in label.split('#'):
                type = ann[2:]
                if 'B-' in ann:
                    annotations.append(Annotation(type, sid, tid))
                elif 'I-' in ann:
                    for _ann in reversed(annotations):
                        if type == _ann.annotation:
                            _ann.add_id(tid)
                            break
            tid += 1
    return annotations


def get_lengths(y_true):
    lengths = []
    for y in np.argmax(y_true, -1):
        try:
            i = list(y).index(0)
        except ValueError:
            i = len(y)
        lengths.append(i)
    return lengths


class NERSequence(Sequence):

    def __init__(self, x, y, batch_size=1, preprocess=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.preprocess = preprocess

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        return self.preprocess(batch_x, batch_y)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)


class NestedReport:
    def __init__(self, *args):
        if len(args) == 2:
            self.report = self.nested_classification_report(*args)

        if len(args) == 1:
            self.report = self.cv_nested_classification_report(*args)

    def __str__(self):
        return self.report

    def support_counter(self, annotations):
        ann_count = Counter()
        anns = [str(ann) for ann in annotations]
        ann_count.update(anns)
        return ann_count

    def husk_annotations(self, annotations, annotation):
        husked_annotations = []
        for ann in annotations:
            if str(ann) == str(annotation):
                husked_annotations.append(ann)
        return husked_annotations

    def cv_nested_classification_report(self, annotations_TP_FP_FN_support):
        TP_counter = Counter()
        FP_counter = Counter()
        FN_counter = Counter()
        supp_counter = Counter()
        report = '{:<22}{:>8}{:>8}{:>8}{:>10}{:>10}{:>10}{:>10}\n'.format('annotation', 'TP', 'FP', 'FN', 'precision',
                                                                          'recall', 'f1-score', 'support')

        for annotation, TP, FP, FN, support in annotations_TP_FP_FN_support:
            TP_counter.update({annotation: TP})
            FP_counter.update({annotation: FP})
            FN_counter.update({annotation: FN})
            supp_counter.update({annotation: support})

        for key, value in sorted(supp_counter.items()):
            if key == 'TOTAL':
                continue
            TP = TP_counter[key]
            FP = FP_counter[key]
            FN = FN_counter[key]

            TP_total += TP
            FP_total += FP
            FN_total += FN
            supp_total += supp_counter[key]

            precision, recall, f1_score = self.calc_p_r_f1(TP, FP, FN)
            data = [str(key), TP, FP, FN, precision * 100, recall * 100, f1_score * 100, supp_counter[key]]
            report += '{0[0]:<22}{0[1]:>8}{0[2]:>8}{0[3]:>8}{0[4]:>10.2f}{0[5]:>10.2f}{0[6]:>10.2f}{0[7]:>10}\n'.format(
                data)
        precision_total, recall_total, f1_score_total = self.calc_p_r_f1(TP_total, FP_total, FN_total)
        total = ["TOTAL"] + [TP_total] + [FP_total] + [FN_total] + [precision_total*100] + [recall_total*100] + [f1_score_total*100] + [supp_total]
        report += '\n'
        report += '{0[0]:<22}{0[1]:>8}{0[2]:>8}{0[3]:>8}{0[4]:>10.2f}{0[5]:>10.2f}{0[6]:>10.2f}{0[7]:>10}\n'.format(
            total)
        return report

    def nested_classification_report(self, true_labels, predicted_labels):
        true_annotations = wrap_annotations(true_labels)
        predicted_annotations = wrap_annotations(predicted_labels)
        support = self.support_counter(true_annotations)
        report = '{:<22}{:>8}{:>8}{:>8}{:>10}{:>10}{:>10}{:>10}\n'.format('annotation', 'TP', 'FP', 'FN', 'precision',
                                                                                'recall', 'f1-score', 'support')
        total = [0, 0, 0]

        for key, value in sorted(support.items()):
            true_anns = self.husk_annotations(true_annotations, key)
            predicted_anns = self.husk_annotations(predicted_annotations, key)
            data = self.label_classification_report(key, set(true_anns), set(predicted_anns), value)
            total[0] += data[1]
            total[1] += data[2]
            total[2] += data[3]
            report += '{0[0]:<22}{0[1]:>8}{0[2]:>8}{0[3]:>8}{0[4]:>10.2f}{0[5]:>10.2f}{0[6]:>10.2f}{0[7]:>10}\n'.format(
                data)

        TP = total[0]
        FP = total[1]
        FN = total[2]
        precision, recall, f1_score = self.calc_p_r_f1(TP, FP, FN)
        total = ["TOTAL"] + [TP] + [FP] + [FN] + [precision*100] + [recall*100] + [f1_score*100] + [sum(support.values())]
        report += '\n'
        report += '{0[0]:<22}{0[1]:>8}{0[2]:>8}{0[3]:>8}{0[4]:>10.2f}{0[5]:>10.2f}{0[6]:>10.2f}{0[7]:>10}\n'.format(total)
        return report

    def label_classification_report(self, annotation, true_annotations, predicted_annotations, support):
        TP = true_annotations & predicted_annotations
        FP = predicted_annotations - true_annotations
        FN = true_annotations - predicted_annotations
        precision, recall, f1_score = self.calc_p_r_f1(len(TP), len(FP), len(FN))
        data = [str(annotation), len(TP), len(FP), len(FN), precision*100, recall*100, f1_score*100, support]
        return data

    def calc_p_r_f1(self, TP, FP, FN):
        try:
            recall = TP / (TP + FN)
        except ZeroDivisionError:
            recall = 0.00
        try:
            precision = TP / (TP + FP)
        except ZeroDivisionError:
            precision = 0.00
        try:
            f1_score = (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1_score = 0.00
        return precision, recall, f1_score


class Vocabulary(object):
    """A vocabulary that maps tokens to ints (storing a vocabulary).

    Attributes:
        _token_count: A collections.Counter object holding the frequencies of tokens
            in the model used to build the Vocabulary.
        _token2id: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        _id2token: A list of token strings indexed by their numerical identifiers.
    """

    def __init__(self, max_size=None, lower=False, unk_token=True, specials=('<pad>',)):
        """Create a Vocabulary object.

        Args:
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            lower: boolean. Whether to convert the texts to lowercase.
            unk_token: boolean. Whether to add unknown token.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary. Default: ('<pad>',)
        """
        self._max_size = max_size
        self._lower = lower
        self._unk = unk_token
        self._token2id = {token: i for i, token in enumerate(specials)}
        self._id2token = list(specials)
        self._token_count = Counter()

    def __len__(self):
        return len(self._token2id)

    def add_token(self, token):
        """Add token to vocabulary.

        Args:
            token (str): token to add.
        """
        token = self.process_token(token)
        self._token_count.update([token])

    def add_documents(self, docs):
        """Update dictionary from a collection of documents. Each document is a list
        of tokens.

        Args:
            docs (list): documents to add.
        """
        for sent in docs:
            sent = map(self.process_token, sent)
            self._token_count.update(sent)

    def doc2id(self, doc):
        """Get the list of token_id given doc.

        Args:
            doc (list): document.

        Returns:
            list: int id of doc.
        """
        doc = map(self.process_token, doc)
        return [self.token_to_id(token) for token in doc]

    def id2doc(self, ids):
        """Get the token list.

        Args:
            ids (list): token ids.

        Returns:
            list: token list.
        """
        return [self.id_to_token(idx) for idx in ids]

    def build(self):
        """
        Build vocabulary.
        """
        token_freq = self._token_count.most_common(self._max_size)
        idx = len(self.vocab)
        for token, _ in token_freq:
            self._token2id[token] = idx
            self._id2token.append(token)
            idx += 1
        if self._unk:
            unk = '<unk>'
            self._token2id[unk] = idx
            self._id2token.append(unk)

    def process_token(self, token):
        """Process token before following methods:
        * add_token
        * add_documents
        * doc2id
        * token_to_id

        Args:
            token (str): token to process.

        Returns:
            str: processed token string.
        """
        if self._lower:
            token = token.lower()

        return token

    def token_to_id(self, token):
        """Get the token_id of given token.

        Args:
            token (str): token from vocabulary.

        Returns:
            int: int id of token.
        """
        token = self.process_token(token)
        return self._token2id.get(token, len(self._token2id) - 1)

    def id_to_token(self, idx):
        """token-id to token (string).

        Args:
            idx (int): token id.

        Returns:
            str: string of given token id.
        """
        return self._id2token[idx]

    @property
    def vocab(self):
        """Return the vocabulary.

        Returns:
            dict: get the dict object of the vocabulary.
        """
        return self._token2id

    @property
    def reverse_vocab(self):
        """Return the vocabulary as a reversed dict object.

        Returns:
            dict: reversed vocabulary object.
        """
        return self._id2token
