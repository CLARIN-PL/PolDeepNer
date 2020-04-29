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

# https://www.researchgate.net/publication/336666695_Fine-Grained_Named_Entity_Recognition_for_Polish_using_Deep_Learning
path_model_pprai = os.path.join(path_model, "pprai")
pretrained_models['n82-pprai'] = [
    PretrainedModelLoader(os.path.join(path_model_pprai, "poldeepner-kpwr-n82-ft-kgr10-e027"),
                          "ft:" + os.path.join(path_model, "kgr10.plain.skipgram.dim300.neg10.bin")),
    PretrainedModelLoader(os.path.join(path_model_pprai, "poldeepner-kpwr-n82-ccmaca-e040"),
                          "ft:" + os.path.join(path_model, "pl.deduped.maca.skipgram.300.mc10.bin")),
    PretrainedModelLoader(os.path.join(path_model_pprai, "poldeepner-kpwr-n82-ftcc-e046"),
                          "ft:" + os.path.join(path_model, "cc.pl.300.bin"))
]
pretrained_models['n82-ft-kgr10'] = [
    PretrainedModelLoader(os.path.join(path_model_pprai, "poldeepner-kpwr-n82-ft-kgr10-e027"),
                          "ft:" + os.path.join(path_model, "kgr10.plain.skipgram.dim300.neg10.bin")),
]
pretrained_models['n82-ft-ccmaca'] = [
    PretrainedModelLoader(os.path.join(path_model_pprai, "poldeepner-kpwr-n82-ccmaca-e040"),
                          "ft:" + os.path.join(path_model, "pl.deduped.maca.skipgram.300.mc10.bin")),
]
pretrained_models['n82-ft-cc'] = [
    PretrainedModelLoader(os.path.join(path_model_pprai, "poldeepner-kpwr-n82-ftcc-e046"),
                          "ft:" + os.path.join(path_model, "cc.pl.300.bin")),
]

# https://www.researchgate.net/publication/328429192_Recognition_of_Named_Entities_for_Polish-Comparison_of_Deep_Learning_and_Conditional_Random_Fields_Approaches
path_model_poleval18 = os.path.join(path_model, "poleval18")
pretrained_models['nkjp-poleval18'] = [
    PretrainedModelLoader(os.path.join(path_model_poleval18, "poldeepner-nkjp-ftcc-bigru"),
                          "ft:" + os.path.join(path_model, "cc.pl.300.bin")),
    PretrainedModelLoader(os.path.join(path_model_poleval18, "poldeepner-nkjp-ftkgr10plain-lstm"),
                          "ft:" + os.path.join(path_model, "kgr10-plain-sg-300-mC50.bin")),
    PretrainedModelLoader(os.path.join(path_model_poleval18, "poldeepner-nkjp-ftkgr10orth-bigru"),
                          "ft:" + os.path.join(path_model, "kgr10_orths.vec.bin"))
]


def load_pretrained_model(name='nkjp-poleval18'):
    if name in pretrained_models:
        loaders = pretrained_models[name]
        return [loader.load() for loader in loaders]
    else:
        raise Exception("Unknown model: %s. Known models: %s" % (name, get_ptetrained_model_names()))


def get_ptetrained_model_names():
    return pretrained_models.keys()
