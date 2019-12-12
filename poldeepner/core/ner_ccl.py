import os
import corpus2
from corpus_ccl import cclutils, token_utils, corpus_object_utils

import argparse

from poldeepner import PolDeepNer
from pretrained import load_pretrained_model


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='CCL annotation mode.')
    parser.add_argument('-m', required=False, metavar='name', help='model name', default='n82-ft-kgr10')
    parser.add_argument('--fileindex', required=True, metavar='corpus_files.txt', help='File with corpus files paths.')
    return parser.parse_args()


def documents(fileindex):
    with open(fileindex, 'r', encoding='utf-8') as f:
        paths = [line.strip() for line in f if os.path.exists(line.strip())]
    return (cclutils.read_ccl(path) for path in paths)


def sentences(document):
    return (sentence
            for paragraph in document.paragraphs()
            for sentence in paragraph.sentences())


def annotate(fileindex, ner):
    for document in documents(fileindex):
        for sentence in sentences(document):
            try:
                tokens = [(t, t.orth_utf8()) for t in sentence.tokens()]
                tokens, orths = zip(*tokens)
                labels = ner.process_sentence(orths)

                idx = 1
                for t, l in zip(tokens, labels):
                    if 'B-' in l:
                        idx += 1
                        token_utils.set_annotation_for_token(sentence, t, 'NE', idx)
                    elif 'I-' in l:
                        token_utils.set_annotation_for_token(sentence, t, 'NE', idx)

            except Exception as e:
                print("Failed to process the text due the following error: %s" % e)

        old_ccl = document.path().split(';')[0]
        new_ccl = old_ccl.replace('.xml', '.ner.xml')
        cclutils.write_ccl(document, new_ccl)


def main(argv=None):
    args = get_args(argv)

    print("\nLoading the NER model ...")
    model = load_pretrained_model(args.m)
    ner = PolDeepNer(model)
    print("NER model loaded.")

    print("Annotating ...")
    annotate(args.fileindex, ner)
    print("Annotation finished.")


if __name__ == "__main__":
    main()

