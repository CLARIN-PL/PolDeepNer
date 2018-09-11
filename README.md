PolDeepNer
==========

About
-----

*PolDeepNer* recognizes mentions of named entities in plain text using deep learning techniques.  

Contributors
------------

* Michał Marcińczuk <michal.marcinczuk@pwr.edu.pl>
* Jan Kocoń <jan.kocon@pwr.edu.pl>
* Michał Gawor 


Installation
------------

### Preparation

Download and/or unpack models (word embeddings, NN pre-trained models):
```bash
./download_unpack.sh
```

### Using Docker

Build Docker image:
```bash
./build.sh
```

Run Docker image in the interactive mode:
```bash
./run.sh
```

### Using virtual environment

```bash
sudo apt-get install python3-pip python3-dev python-virtualenv
sudo pip install -U pip
virtualenv --system-site-packages -p python3.5 venv
source venv/bin/activate
pip install -U pip

pip install seqeval
pip install keras
pip install tensorflow-gpu
pip install git+https://www.github.com/keras-team/keras-contrib.git
pip install cython
pip install pyfasttext
pip install fasttext
pip install sklearn
pip install python-dateutil
```


Evaluation (using Docker)
----------

### Process set of document and generate a json file

The evalation corpus in json format was split into a set of separate files. 
Then each of the documents was tagged using WCRFT tagger and four files were generated:
* NAME.txt — file content,
* NAME.ini — ini file with document original name,
* NAME.xml — ccl file containing tagging output,
* NAME.iob — ccl file converted to iob format. 

Run Docker

```bash
./run.sh
```

Generate json file for the evaluation corpus:

```bash
python3.6 core/process_poleval.py \
            -i data/poleval2018ner-data/index_iob.list \
            -o data/poldeepner-output.json \
            -m model
```

Run official evaluation scripts:

```bash
python3.6 core/poleval_ner_test.py \
             -g data/POLEVAL-NER_GOLD.json \
             -u data/poldeepner-output.json
```

Output (to update):
```bash
OVERLAP precision: 0.915 recall: 0.806 F1: 0.857 
EXACT precision: 0.860 recall: 0.757 F1: 0.805 
Final score: 0.847
```


Training
--------
Run virtual environment.

```bash
source venv/bin/activate
```

Train each model separately. The training will replace the pre-trained models. 
```bash
python poldeepner/core/trainmodel.py \
              -i poldeepner/data/nkjp-nested-simplified-v2.iob \
              -t poldeepner/data/nkjp-nested-simplified-v2.iob \
              -f poldeepner/model/cc.pl.300.bin \
              -m poldeepner/model/poldeepner-nkjp-ftcc-bigru \
              -e 15 -n GRU
                          
python poldeepner/core/trainmodel.py \
              -i poldeepner/data/nkjp-nested-simplified-v2.iob \
              -t poldeepner/data/nkjp-nested-simplified-v2.iob \
              -f poldeepner/model/cc.pl.300.bin \
              -m poldeepner/model/poldeepner-nkjp-ftcc-bilstm \
              -e 15 -n LSTM
                          
python poldeepner/core/trainmodel.py \
              -i poldeepner/data/nkjp-nested-simplified-v2.iob \
              -t poldeepner/data/nkjp-nested-simplified-v2.iob \
              -f poldeepner/model/kgr10-plain-sg-300-mC50.bin \
              -m poldeepner/model/poldeepner-nkjp-ftkgr10plain \
              -e 15 -n LSTM
```

After every epoch a table with accuracy of the current model will be printed. Below is a sample table after the first epoch.

```bash
2677/2677 [==============================] - 234s 87ms/step - loss: 0.0762
 - f1: 74.68
annotation                  TP      FP      FN precision    recall  f1-score   support
date                      3365     647    1133     83.87     74.81     79.08      4498
geogName                  1102     417    3040     72.55     26.61     38.93      4142
orgName                   6714    2867    4444     70.08     60.17     64.75     11158
persName                 16165    2495    3773     86.63     81.08     83.76     19938
persName_addName             0       0     949      0.00      0.00      0.00       949
persName_forename        11271    1138    1683     90.83     87.01     88.88     12954
persName_surname         11351    2035    1266     84.80     89.97     87.31     12617
placeName                    0       0     368      0.00      0.00      0.00       368
placeName_bloc               0       0     100      0.00      0.00      0.00       100
placeName_country         6172    1437    1324     81.11     82.34     81.72      7496
placeName_district           0       0     279      0.00      0.00      0.00       279
placeName_region             0       0     760      0.00      0.00      0.00       760
placeName_settlement      5796    2055    1984     73.82     74.50     74.16      7780
time                       147      45     417     76.56     26.06     38.89       564

TOTAL                    62083   13136   21520     82.54     74.26     78.18     83603
```


License
-----
Copyright (C) Wrocław University of Science and Technology (PWr), 2010-2018. All rights reserved.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.