import json
import argparse
import os
import codecs

from iob import load_data_and_labels
from poldeepner import PolDeepNer
from utils import wrap_annotations


def get_id(ini_file):
    for line in codecs.open(ini_file, "r", "utf8"):
        if 'id = ' in line:
            return line.replace('id = ', '')


def align_tokens_to_text(sentences, text):
    offsets = []
    tid = 0
    for s in sentences:
        for t in s:
            start = text.find(t, tid)
            if start == -1:
                raise Exception("Could not align tokens to text")
            end = start + len(t)
            offsets.append((start, end))
            tid = end
    return offsets


def get_poleval_dict(doc_id, text, sentences, labels):
    ''' Returns PolEval dict
    {
        text:
        id:
        answers:
    }
    Note that arguments it takes is FILE, PATH, FILE as utils.load_data_and_labels opens file itself
    '''
    annotations = wrap_annotations(labels)
    offsets = align_tokens_to_text(sentences, text)
    answers = []
    for an in annotations:
        begin = offsets[an.token_ids[0]][0]
        end = offsets[an.token_ids[-1]][1]
        orth = text[begin:end]
        answers.append("%s %d %d\t%s" % (an.annotation.replace("-", "_"), begin, end, orth))
    return ({
        'text': text,
        'id': doc_id,
        'answers': "\n".join(answers)
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert set of IOB, TXT and INI files into a single json file in PolEval 2018 NER format')
    parser.add_argument('-i', required=True, metavar='PATH', help='path to a file with a list of files')
    parser.add_argument('-o', required=True, metavar='PATH', help='path to a json output file')
    parser.add_argument('-m', required=True, metavar='PATH', help='path to the model')

    args = parser.parse_args()
    path = args.i

    parent = os.path.dirname(path)

    ner = PolDeepNer(args.m)

    dict_list = []
    paths = codecs.open(path, "r", "utf8").readlines()
    paths_count = len(paths)
    for n, rel_path in enumerate(paths):
        abs_path = os.path.abspath(os.path.join(parent, rel_path.strip()))
        namext = os.path.basename(abs_path)
        name = os.path.splitext(namext)[0]
        path = os.path.dirname(abs_path)

        text = codecs.open(os.path.join(path, name + ".txt"), "r", "utf8").read()
        doc_id = get_id(os.path.join(path, name + ".ini"))
        print("%d from %d: %s" % (n, paths_count, doc_id))

        sentences, _ = load_data_and_labels(os.path.join(path, name + '.iob'))
        labels = ner.process_document(sentences)

        dict_list.append(get_poleval_dict(doc_id, text, sentences, labels))

    codecs.open(args.o, "w", "utf8").write(json.dumps(dict_list))
