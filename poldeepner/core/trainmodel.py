import argparse
import os

from tensorflow.python.client import device_lib

from load_data import load_data
from wrapper import Sequence
from embedding_wrapper import load_embedding

parser = argparse.ArgumentParser(description='Process IOB file, recognize NE and save the output to another IOB file.')
parser.add_argument('-i', required=True, metavar='PATH', help='input train file .iob .xml or index file')
parser.add_argument('-t', required=False, metavar='PATH', help='input test IOB file')
parser.add_argument('-f', required=True, metavar='PATH', help='path to .vec keyed vector (gensim) or .bin embedding (fasttext)')
parser.add_argument('-m', required=True, metavar='PATH', help='path to a folder in which the model will be saved')
parser.add_argument('-n', required=True, metavar='nn_type', help='type of NN: GRU or LSTM', default='GRU')
parser.add_argument('-e', required=True, default=32, type=int, metavar='num', help='number of epoches')
parser.add_argument('-g', required=True, nargs='+', help='which GPUs to use')
parser.add_argument('-C', action='store_true', help='append char embedding created from training data')

args = parser.parse_args()
model = args.m

gpus = ''
for gpu_nb in args.g:
    gpus += str(gpu_nb) + ' '
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
print(device_lib.list_local_devices())


# _____LOAD DATA______
x_train, y_train = load_data(args.i)
print("Train: %d" % len(x_train))

if args.t:
    x_test, y_test = load_data(args.t)
    print("Test : %d" % len(x_test))
else:
    x_test, y_test = None, None

# _____LOAD EMBEDDING_____
embedding = load_embedding(args.f)


# _____BUILD AND TRAIN MODEL_____
m = Sequence(embedding, use_char=args.C, nn_type=args.n)
m.fit(x_train, y_train, x_test, y_test, epochs=args.e, batch_size=32)

# _____SAVE MODEL_____
model = os.path.join(model, 'poldeepner-' + embedding.name)

if not os.path.exists(model):
    os.makedirs(model)

m.save(model)
