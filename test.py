from load_data import load_data
from embedding_wrapper import load_embedding
from wrapper import Sequence
import argparse


parser = argparse.ArgumentParser(description='Process IOB file, recognize NE and save the output to another IOB file.')
parser.add_argument('-i', required=True, metavar='PATH', help='input IOB file')

args = parser.parse_args()

if __name__ == '__main__':
    embedding = load_embedding('/PolDeepNer/poldeepner/model/kgr10.plain.skipgram.dim300.neg10.bin')
    model = Sequence.load('/PolDeepNer/poldeepner/model/poldeepner-kgr10.plain.skipgram.dim300.neg10.bin', embedding)
    model.predict_to_iob(args.i, '/PolDeepNer/test_result.iob')
