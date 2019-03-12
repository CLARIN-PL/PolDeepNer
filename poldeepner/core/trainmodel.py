import argparse
import os

from tensorflow.python.client import device_lib

import iob
from wrapper import Sequence

parser = argparse.ArgumentParser(description='Process IOB file, recognize NE and save the output to another IOB file.')
parser.add_argument('-i', required=True, metavar='PATH', help='input train IOB file')
parser.add_argument('-t', required=True, metavar='PATH', help='input test IOB file')
parser.add_argument('-f', required=True, metavar='PATH', help='path to a FastText bin file with embeddings')
parser.add_argument('-m', required=True, metavar='PATH', help='path to a folder in which the model will be saved')
parser.add_argument('-n', required=True, metavar='nn_type', help='type of NN: GRU or LSTM', default='GRU')
parser.add_argument('-e', required=True, default=32, type=int, metavar='num', help='number of epoches')
parser.add_argument('-s', required=False, default=300, type=int, metavar='num', help='size of the input')
parser.add_argument('-g', required=True, inargs='+', help='which GPUs to use')

args = parser.parse_args()
model = args.m

gpus = ''
for gpu_nb in args.g:
    gpus += str(gpu_nb) + ' '

os.environ["CUDA_VISIBLE_DEVICES"] = gpus
print(device_lib.list_local_devices())

if not os.path.exists(model):
    os.mkdir(model)

model_weights = os.path.join(model, "weights.pkl")
model_params = os.path.join(model, "params.pkl")
model_preprocessor = os.path.join(model, "preprocessor.pkl")

x_train, y_train = iob.load_data_and_labels(args.i)
x_test, y_test = iob.load_data_and_labels(args.t)
print("Train: %d" % len(x_train))
print("Test : %d" % len(x_test))

m = Sequence(args.f, use_char=False, nn_type=args.n, input_size=args.s)
m.fit(x_train, y_train, args.f, x_test, y_test, epochs=args.e, batch_size=32)
m.save(model_weights, model_params, model_preprocessor)

