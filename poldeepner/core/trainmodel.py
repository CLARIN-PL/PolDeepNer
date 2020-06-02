import argparse
import os

from tensorflow.python.client import device_lib

from load_data import load_data
from wrapper import Sequence

parser = argparse.ArgumentParser(description='Process IOB file, recognize NE and save the output to another IOB file.')
parser.add_argument('-i', required=True, metavar='PATH', help='input train file .iob .xml or index file')
parser.add_argument('-t', required=False, metavar='PATH', help='input test IOB file')
parser.add_argument('-f', required=True, metavar='PATH',
                    help='embedding in form of type:path, where type=ft|elmo|elmo-avg|elmo-concat')
parser.add_argument('-m', required=True, metavar='PATH', help='path to a folder in which the model will be saved')
parser.add_argument('-n', required=True, metavar='nn_type', help='type of NN: GRU or LSTM', default='GRU')
parser.add_argument('-e', required=True, default=32, type=int, metavar='num', help='number of epoches')
parser.add_argument('-g', nargs='+', help='which GPUs to use')
parser.add_argument('-C', action='store_true', help='use char embedding built from training data')
parser.add_argument('-p', required=False, metavar='PATH', help='input pretrained model for transfer learning')

args = parser.parse_args()
model = args.m

if args.g:
    os.environ["CUDA_VISIBLE_DEVICES"] = " ".join(map(str, args.g))

print(device_lib.list_local_devices())

# _____LOAD DATA______
x_train, y_train = load_data(args.i)
print("Train: %d" % len(x_train))

x_test, y_test = None, None
if args.t:
    x_test, y_test = load_data(args.t)
    print("Test : %d" % len(x_test))


# _____BUILD AND TRAIN MODEL_____

m = Sequence(args.f, use_char=args.C, nn_type=args.n, transfer_model=args.p)
m.fit(x_train, y_train, x_test, y_test, epochs=args.e, batch_size=32)

# _____SAVE MODEL_____
if not os.path.exists(model):
    os.makedirs(model)
m.save(model)
