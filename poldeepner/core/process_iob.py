"""
Usage: python process_iob.py

"""
import argparse
import codecs

from poldeepner import PolDeepNer

parser = argparse.ArgumentParser(description='Process IOB file, recognize NE and save the output to another IOB file.')
parser.add_argument('-i', required=True, metavar='PATH', help='input IOB file')
parser.add_argument('-m', required=True, metavar='PATH', help='path to the model')
parser.add_argument('-o', required=True, metavar='PATH', help='output IOB file')

args = parser.parse_args()

ner = PolDeepNer(args.m)


def process_file(input, output, ner):
    with codecs.open(input, "r", "utf8") as f:
        fo = codecs.open(output, "w", "utf8")
        lines, words = [], []
        for line in f:
            line = line.rstrip()
            if "-DOCSTART " in line:
                fo.write(line + "\n")
                pass
            elif line:
                cols = line.split('\t')
                words.append(cols[0])
                lines.append("\t".join(cols[:-1]))
            else:
                labels = ner.process_sentence(words)
                for pair in zip(lines, labels):
                    fo.write("%s\t%s\n" % pair)
                lines, words = [], []


process_file(args.i, args.o, ner)
