import os
from wrapper import Sequence


class PretrainedModelLoader:

    def __init__(self, path_folder, embeddings):
        self.path_folder = path_folder
        self.embeddings = embeddings

    def load(self):
        return Sequence.load(self.path_folder, self.embeddings)


path_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
path_model = os.path.join(path_root, "model")

pretrained_models = {}

path_model_pprai = os.path.join(path_model, "pprai")
pretrained_models['n82'] = [
    PretrainedModelLoader(os.path.join(path_model_pprai, "poldeepner-kpwr-n82-ft-kgr10-e027"),
                          "ft:" + os.path.join(path_model, "kgr10.plain.skipgram.dim300.neg10.bin")),
    PretrainedModelLoader(os.path.join(path_model_pprai, "poldeepner-kpwr-n82-ccmaca-e040"),
                          "ft:" + os.path.join(path_model, "pl.deduped.maca.skipgram.300.mc10.bin")),
    PretrainedModelLoader(os.path.join(path_model_pprai, "poldeepner-kpwr-n82-ftcc-e046"),
                          "ft:" + os.path.join(path_model, "cc.pl.300.bin"))
]


def load_pretrained_model(name='n82'):
    if name in pretrained_models:
        loaders = pretrained_models[name]
        return [loader.load() for loader in loaders]
    else:
        raise Exception("Unknown model name: %s. Known models: " % (name, ", ".join(pretrained_models.keys())))
