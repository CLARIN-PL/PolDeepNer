from operator import itemgetter
import os

from wrapper import Sequence

class PolDeepNer:

    def __init__(self, model):
        """

        :param model: Path to a folder with a model to deserialize
        """
        self.models = []
        self.models.append(Sequence.load(os.path.join(model, "poldeepner-nkjp-ftcc-bilstm"), os.path.join(model, "cc.pl.300.bin")))
        self.models.append(Sequence.load(os.path.join(model, "poldeepner-nkjp-ftcc-bigru"), os.path.join(model, "cc.pl.300.bin")))
        self.models.append(Sequence.load(os.path.join(model, "poldeepner-nkjp-ftkgr10plain"), os.path.join(model, "kgr10-plain-sg-300-mC50.bin")))

    def process_sentence(self, sentence):
        """
        Process a single sentence and return an array with token labels.
        :param sentence: An array of the sentence words
        :return: An array of token labels
        """
        predictions = []
        for m in self.models:
            predictions.append(m.predict_sentence(sentence))
        final_prediction = []
        for n in range(0, len(sentence)):
            votes = {}
            for m in range(0, len(predictions)):
                item = predictions[m][n]
                votes[item] = (votes[item] if item in votes else 0) + 1
            final_prediction.append(sorted(votes.items(), key=itemgetter(1), reverse=True)[0][0])
        return final_prediction

    def process_document(self, sentences):
        """
        Process a document as an array of sentences. Each sentence is an array of words.
        :param sentences: An array of sentence
        :return: An array of token lables for each sentence
        """
        output = []
        for sentence in sentences:
            output.append(self.process_sentence(sentence))
        return output
