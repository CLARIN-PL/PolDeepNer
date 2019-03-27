#! /usr/bin/python3

import logging
import os
from pathlib import Path
import subprocess
import tempfile

import nlp_ws
from poldeepner.core.process_file import process_file

_log = logging.getLogger(__name__)


def check_models_paths(models_embeddings):
    for model_path, embedding_path in models_embeddings.items():
        if not (os.path.exists(model_path) and os.path.exists(embedding_path)):
            return False
    return True


class PolDeepNerWorker(nlp_ws.NLPWorker):
    def process(self, input_path, task_options, output_path):
        if task_options is None:
            task_options = {'models': {'./poldeepner/model/poldeepner-kgr10.plain.skipgram.dim300.neg10.bin':
                                       './poldeepner/model/kgr10.plain.skipgram.dim300.neg10.bin'}}
        elif 'models' not in task_options:
            raise WrongTaskOptions('Models not in task options: ' + str(task_options))

        elif not check_models_paths(task_options['models']):
            raise WrongTaskOptions('Wrong paths to models: ' + str(task_options['models']))

        # Create tmp file where toki output will be stored
        (iobfd, iob_file_path) = tempfile.mkstemp(suffix='.xml', dir='./tmp')

        # Use liner to convert input ccl format to iob
        p = subprocess.Popen('liner-cli -i ccl -f ' + input_path + ' -o iob -t ' + iob_file_path)
        p.wait()

        # Process .iob file
        process_file(iob_file_path, output_path, task_options['models'])
        os.remove(iob_file_path)


class WrongTaskOptions(Exception):
    def __init__(self, message):
        super(WrongTaskOptions, self).__init__(message)


if __name__ == '__main__':
    nlp_ws.NLPService.main(PolDeepNerWorker)

