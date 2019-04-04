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
        models = {'./../model/poldeepner-kgr10.plain.skipgram.dim300.neg10.bin':
                  './../model/kgr10.plain.skipgram.dim300.neg10.bin'}
        ner = PolDeepNer(list(models.keys()), list(models.values()))
    x, _, ext_data = load_data(input_path)
    y_pred = ner.process_document(x)
    print(y_pred)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, 'w') as output_file:
        for sentence, labels, ext_sent in zip(x, y_pred, ext_data):
            for token, label, ext_cols in zip(sentence, labels, ext_sent):
                line = token
                for ext_col in ext_cols:
                    line += '\t' + ext_col
                if label != '':
                    line += '\t' + label + '\n'
                else:
                    line += '\tO\n'
                print(line)
                output_file.write(line)
            output_file.write('\n')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    process_file('/mnt/big_one/gawor/data/nkjp-ratio4v2-nested-simplified-seta.iob', '/mnt/big_one/gawor/data/result.txt')
