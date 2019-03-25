import argparse
import configparser
import codecs
import os

from load_data import load_data
from poldeepner import PolDeepNer


def process_file(input_path, output_path, models=None):
    """function for predicting Named Entities using PolDeepNer
    :param input_path: path to .iob, .xml (CCL format), .tok or index file
    :param output_path: path to file where prediction output in .iob format will be stored
    :param models: dict {model_dir_path: embedding_file_path}
    """

    if models is not None:
        ner = PolDeepNer(list(models.keys()), list(models.values()))
    else:
        models = {'./../data/': './../data/cc.pl.300.bin',
                  './../dat/': './../data/',
                  '': ''}
        ner = PolDeepNer(list(models.keys()), list(models.values()))
    x, _ = load_data(input_path)
    y_pred = ner.process_document(x)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, 'w') as output_file:
        for sentence, labels in zip(x, y_pred):
            for token, label in zip(sentence, labels):
                output_file.write(token + '\t' + label + '\n')
            output_file.write('\n')
