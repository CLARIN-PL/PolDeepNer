import os
import warnings
import iob

warnings.filterwarnings('ignore', category=FutureWarning)


def parse_args():
    import argparse

    root = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(root, "..", "data")
    path_eval = os.path.join(path_data, "kpwr-ner-n82-test.iob")

    parser = argparse.ArgumentParser(description='Evaluate given model against annotated document in IOB format.')
    parser.add_argument('--model', metavar='name/path', help='model name or path to model', default='n82-ft-kgr10')
    parser.add_argument('--embeddings', metavar='PATH',
                        help='embedding in form of type:path, where type=ft|elmo|elmo-avg|elmo-concat')
    parser.add_argument('--input', metavar='path', help='path to IOB data to evaluate', default=path_eval)
    return parser.parse_args()


def main(args):
    from poldeepner import PolDeepNer
    from pretrained import load_pretrained_model
    from utils import NestedReport
    from wrapper import Sequence

    try:
        model = [Sequence.load(args.model, args.embeddings)] if args.embeddings else load_pretrained_model(args.model)
        ner = PolDeepNer(model)

        label_true, label_pred = [], []
        x_test, y_test = iob.load_data_and_labels(args.input)
        n = 0
        for x, y in zip(x_test, y_test):
            pred = ner.process_sentence(x)
            label_true.append(y)
            label_pred.append(pred)
            if n % 1000 == 0:
                print("Sentences processed: %d / %d" % (n, len(y_test)))
            n += 1
        print("Sentences processed: %d / %d" % (n, len(y_test)))

        report = NestedReport(label_true, label_pred)
        print(str(report))

    except Exception as e:
        print("[ERROR] %s" % str(e))


if __name__ == "__main__":
    cli_args = parse_args()
    print("Command Line Args:", cli_args)
    main(cli_args)
