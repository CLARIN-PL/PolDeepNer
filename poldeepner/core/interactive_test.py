import argparse

import nltk, re, pprint
from nltk import word_tokenize

from poldeepner import PolDeepNer
from pretrained import load_pretrained_model
from process_poleval import align_tokens_to_text
from utils import wrap_annotations

parser = argparse.ArgumentParser(description='Interactive mode')
parser.add_argument('-m', required=True, metavar='name', help='model name', default='n82')
args = parser.parse_args()


print("Loading the tokenization model ...")
nltk.download('punkt')

print("Loading the NER model ...")
model = load_pretrained_model(args.m)
ner = PolDeepNer(model)

print("ready.")

while True:
    text = input("Enter text to process: ").strip().replace("\"", "'")

    if len(text) == 0:
        print("closing...")
        break

    try:
        # ToDo: replace with toki or maca.
        tokens = word_tokenize(text)
        labels = ner.process_sentence(tokens)
        offsets = align_tokens_to_text([tokens], text)

        for an in wrap_annotations([labels]):
            begin = offsets[an.token_ids[0]][0]
            end = offsets[an.token_ids[-1]][1]
            orth = text[begin:end]

            print("[%3s:%3s] %-20s %s" % (begin, end, an.annotation, orth))

    except Exception as e:
        print("Failed to process the text due the following error: %s" % e)

