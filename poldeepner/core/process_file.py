"""
Usage: python process_iob.py

"""
import argparse
import codecs

import nltk
from nltk import word_tokenize

from poldeepner import PolDeepNer
from pretrained import load_pretrained_model
from process_poleval import align_tokens_to_text
from utils import wrap_annotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process IOB file, recognize NE and save the output to another IOB file.')
    parser.add_argument('-i', required=True, metavar='PATH', help='path to a plain text')
    parser.add_argument('-m', required=False, metavar='NAME', help='name of a model pack')
    args = parser.parse_args()

    try:
        print("Loading the tokenization model ...")
        nltk.download('punkt')

        print("Loading the NER model ...")
        model = load_pretrained_model(args.m)
        ner = PolDeepNer(model)

        print("ready.")

        text = " ".join(codecs.open(args.i, "r", "utf8").readlines())
        tokens = word_tokenize(text)
        labels = ner.process_sentence(tokens)
        offsets = align_tokens_to_text([tokens], text)

        for an in wrap_annotations([labels]):
            begin = offsets[an.token_ids[0]][0]
            end = offsets[an.token_ids[-1]][1]
            orth = text[begin:end]

            print("[%3s:%3s] %-20s %s" % (begin, end, an.annotation, orth))

    except Exception as e:
        print("[ERROR] %s" % str(e))
