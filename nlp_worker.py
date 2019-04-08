import logging
from operator import itemgetter
import os
from pathlib import Path
import subprocess
import sys
import tempfile,shutil

import nlp_ws
from process_file import process_file
from poldeepner import PolDeepNer
from embedding_wrapper import load_embedding
from wrapper import Sequence

_log = logging.getLogger(__name__)


class PolDeepNerWorker(nlp_ws.NLPWorker):
    def __init__(self):
        embedding = load_embedding('/PolDeepNer/poldeepner/model/kgr10.plain.skipgram.dim300.neg10.bin')
        self.model = Sequence.load('/PolDeepNer/poldeepner/model/poldeepner-kgr10.plain.skipgram.dim300.neg10.bin', embedding)
    def process(self, input_path, task_options, output_path):                	
        #make temp file
        temp_folder = tempfile.mkdtemp()
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)    
            
        new_path=os.path.join(temp_folder, 'data.iob');    
        shutil.copy2(input_path,new_path)    
        print(new_path)
        # Process .iob file
        self.model.predict_to_iob(new_path, output_path)
        
        #remove temp file
        shutil.rmtree(temp_folder)	



class WrongTaskOptions(Exception):
    def __init__(self, message):
        super(WrongTaskOptions, self).__init__(message)


if __name__ == '__main__':
    nlp_ws.NLPService.main(PolDeepNerWorker)

