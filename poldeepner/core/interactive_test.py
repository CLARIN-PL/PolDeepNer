import warnings
import nltk
from nltk import word_tokenize

warnings.filterwarnings('ignore', category=FutureWarning)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Interactive mode')
    parser.add_argument('--model', metavar='name/path', help='model name or path to model', default='n82-ft-kgr10')
    return parser.parse_args()


def run_cli_loop(ner):
    from process_poleval import align_tokens_to_text
    from utils import wrap_annotations

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


def main(args):
    from poldeepner import PolDeepNer
    from pretrained import load_pretrained_model

    try:
        print("Loading the tokenization model ...")
        nltk.download('punkt')

        print("Loading the NER model ...")
        model = load_pretrained_model(args.model)
        ner = PolDeepNer(model)

        print("ready.")
        run_cli_loop(ner)

    except Exception as e:
        print("[ERROR] %s" % str(e))


if __name__ == "__main__":
    cli_args = parse_args()
    print("Command Line Args:", cli_args)
    main(cli_args)

