"""
Usage: python process_iob.py

"""
import argparse
import codecs
import os

from load_data import load_data
from poldeepner import PolDeepNer

parser = argparse.ArgumentParser(description='Process IOB file, recognize NE and save the output to another IOB file.')
parser.add_argument('-i', required=True, metavar='PATH', help='input IOB file')
parser.add_argument('-m', required=True, metavar='PATH', help='path to the model')
parser.add_argument('-o', required=True, metavar='PATH', help='output IOB file')
parser.add_argument('-f', required=True, metavar='PATH', help='path to embedding')

args = parser.parse_args()

ner = PolDeepNer([args.m], [args.f])
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def process_file(input_path, output_path, ner):
    x, _ = load_data(input_path)
    y_pred = ner.process_document(x)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, 'w') as output_file:
        for sentence, labels in zip(x, y_pred):
            for token, label in zip(sentence, labels):
                output_file.write(token + '\t' + label + '\n')
            output_file.write('\n')


process_file(args.i, args.o, ner)
