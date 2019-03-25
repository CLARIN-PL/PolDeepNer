"""
Usage: python process_iob.py

"""
import argparse
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
        models = {'./../data/poldeepner-nkjp-ftcc-bigru/': './../data/cc.pl.300.bin',
                  './../data/poldeepner-nkjp-ftkgr10orth-bigru': './../data/kgr10_orths.vec.bin',
                  './../data/poldeepner-nkjp-ftkgr10plain-lstm': './../data/kgr10-plain-sg-300-mC50.bin'}
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process IOB file, recognize NE and save the output to another IOB file.')
    parser.add_argument('-i', required=True, metavar='PATH', help='input IOB file')
    parser.add_argument('-m', required=False, metavar='PATH', help='path to the model')
    parser.add_argument('-o', required=True, metavar='PATH', help='output IOB file')
    parser.add_argument('-f', required=False, metavar='PATH', help='path to embedding')

    args = parser.parse_args()

    ner = PolDeepNer([args.m], [args.f])
