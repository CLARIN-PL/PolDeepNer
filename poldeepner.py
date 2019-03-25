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
        if 'models' not in task_options:
            task_options['models'] = dict()
        elif not check_models_paths(task_options['models']):
            raise WrongTaskOptions('Wrong paths to models: ' + str(task_options['models']))
        # Create tmp file where toki output will be stored
        (tokfd, tok_file_path) = tempfile.mkstemp(suffix='.tok', dir='./tmp')
        (iobfd, iob_file_path) = tempfile.mkstemp(suffix='.iob', dir='./tmp')
        # Run toki in order to tokenise input .txt file and save it in tmp .tok file
        p = subprocess.Popen('toki-app -f \$orth\\t\$ws\\n < ' + input_file_path + ' > ' + tok_file_path)
        p.wait()

        # Process .tok file
        process_file(tok_file_path, iob_file_path, task_options['models'])
        os.remove(tmp_file_path)


class WrongTaskOptions(Exception):
    def __init__(self, message):
        super(WrongTaskOptions, self).__init__(message)


if __name__ == '__main__':
    nlp_ws.NLPService.main(PolDeepNerWorker)