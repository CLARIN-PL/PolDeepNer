import argparse
import os

from tensorflow.python.client import device_lib

from load_data import load_iob, load_xml, UnsupportedFileFormat
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

args = parser.parse_args()
model = args.m

gpus = ''
for gpu_nb in args.g:
    gpus += str(gpu_nb) + ' '
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
print(device_lib.list_local_devices())

# _____LOAD DATA_____
x_test, y_test = [], []
x_train, y_train = [], []

# Get data from iob file
if args.i.endswith('.iob'):
    x_train, y_train = load_iob(args.i)

# Get data from xml file
elif args.i.endswith('.xml'):
    x_train, y_train = load_xml(args.i)

# Get data from index file
else:
    with open(args.i, 'r') as index_file:
        for index in index_file:
            index = index.replace('\n', '')
            file_path = os.path.join(os.path.dirname(args.i), index)
            # Get data from iob listed in index file
            if index.endswith('.iob'):
                x, y = load_iob(file_path)
                x_train += x
                y_train += y

            # Get data from xml listed in index file
            elif index.endswith('xml'):
                x, y = load_xml(file_path)
                x_train += x
                y_train += y

            else:
                raise UnsupportedFileFormat('Unsupported file format of file: ' + os.path.basename(index))

# Get test data if provided
if args.t:
    # Get test data from iob file
    if args.t.endswith('.iob'):
        x_test, y_test = load_iob(args.t)

    # Get test data from xml file
    elif args.t.endswith('.xml'):
        x_test, y_test = load_xml(args.t)

    # Get test from index file
    else:
        with open(args.t, 'r') as index_file:
            for index in index_file:
                index = index.replace('\n', '')
                file_path = os.path.join(os.path.dirname(args.t), index)

                # Get data from iob listed in index file
                if index.endswith('.iob'):
                    x, y = load_iob(file_path)
                    x_test += x
                    y_test += y

                # Get data from xml listed in index file
                elif index.endswith('.xml'):
                    x, y = load_xml(file_path)
                    x_test += x
                    y_test += y

                else:
                    raise UnsupportedFileFormat('Unsupported file format of file: ' + os.path.basename(index))

print("Train: %d" % len(x_train))
if args.t:
    print("Test : %d" % len(x_test))

# _____LOAD EMBEDDING_____
embedding = load_embedding(args.f)


# _____BUILD AND TRAIN MODEL_____
m = Sequence(embedding, use_char=False, nn_type=args.n)
m.fit(x_train, y_train, x_test, y_test, epochs=args.e, batch_size=32)

# _____SAVE MODEL_____
model = os.path.join(model, 'poldeepner' + embedding.name)

if not os.path.exists(model):
    os.makedirs(model)

m.save(model)
