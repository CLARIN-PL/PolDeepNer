PolDeepNer
==========

About
-----

*PolDeepNer* recognizes mentions of named entities in text using deep learning methods. 
The tool won second place in the [PolEval 2018 Task 2](http://poleval.pl/tasks/) on named entity recognition. 
It contains a pre-trained model trained on the [NKJP corpus](http://clip.ipipan.waw.pl/NationalCorpusOfPolish) 
which recognizes nested annotations of the following types:

![NKJP NER categories](docs/media/nkjp-ner-schema.png)

### Paper

Marcińczuk, Michał; Kocoń, Jan; Gawor, Michał. 
_Recognition of Named Entities for Polish-Comparison of Deep Learning and Conditional Random Fields Approaches_
Ogrodniczuk, Maciej; Kobyliński, Łukasz (Eds.): 
Proceedings of the PolEval 2018 Workshop, pp. 63-73, Institute of Computer Science, 
Polish Academy of Science, Warszawa, 2018.

\[[PDF](https://www.researchgate.net/publication/328429192_Recognition_of_Named_Entities_for_Polish-Comparison_of_Deep_Learning_and_Conditional_Random_Fields_Approaches)\]

<details><summary>[Bibtex]</summary>
<p>

```
@inproceedings{poldeepner2018,
  title     = "Recognition of Named Entities for Polish-Comparison of Deep Learning and Conditional Random Fields Approaches",
  author    = "Marcińczuk, Michał and Kocoń, Jan and Gawor, Michał",
  year      = "2018",
  editor    = "Ogrodniczuk, Maciej and Kobyliński, Łukasz",
  booktitle = "Proceedings of the PolEval 2018 Workshop",
  location  = "Warsaw, Poland",
  pages     = "77--92",
  publisher = "Institute of Computer Science, Polish Academy of Science"
}
```

</p>
</details>

### Credits

BiLSTM and CRF implementation was based on AnaGo (https://github.com/Hironsan/anago)

Contributors
-----------------------------------------------------------

* Michał Marcińczuk <michal.marcinczuk@pwr.edu.pl>
* Jan Kocoń <jan.kocon@pwr.edu.pl>
* Michał Gawor <michal.gawor@pwr.edu.pl>


Requirements
-----------------------------------------------------------
* Python 3.6
* CUDA 10.0+


Installation
-----------------------------------------------------------

### Preparation

Download and/or unpack models (word embeddings, NN pre-trained models):
```bash
sudo apt-get install p7zip-full
```

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
virtualenv -p python3.6 venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```


PolEval 2018 (NKJP NER model)
-----------------------------------------------------------

### Evaluation (using Docker)

#### Process set of document and generate a json file

The evaluation corpus in json format was split into a set of separate files. 
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
python poldeepner/core/process_poleval.py \
            -i poldeepner/data/poleval2018ner-data/index_iob.list \
            -o poldeepner/data/poldeepner-output.json \
            -m nkjp-poleval18
```

Run the official evaluation script:
```bash
python poldeepner/core/poleval_ner_test.py \
             -g poldeepner/data/POLEVAL-NER_GOLD.json \
             -u poldeepner/data/poldeepner-output.json
```

Output:
```bash
OVERLAP precision: 0.918 recall: 0.815 F1: 0.864 
EXACT precision: 0.860 recall: 0.764 F1: 0.809 
Final score: 0.853
```
 

### Processing

#### Interactive test

*Disclaimer:* the current version of the script uses NLTK tokenizer which is not fully compatible with the training data used to train the model.
The final version of the script should use *toki* tokenizer or *WCRFT* tagger. 

Run:

```bash
python poldeepner/core/interactive_test.py
```

Enter text to process:

```bash
Enter text to process:
```

```bash
Enter text to process: XXXIII konwent fanów fantastyki Polcon trwał przez 4 dni: od 12 do 15 sierpnia 2018 na terenie Miasteczka Akademickiego Uniwersytetu Mikołaja Kopernika w Toruniu.
```

Output:
```bash
[ 32: 38] orgName              Polcon
[ 61: 63] date                 12
[ 67: 83] date                 15 sierpnia 2018
[ 95:151] geogName             Miasteczka Akademickiego Uniwersytetu Mikołaja Kopernika
[154:161] placeName_settlement Toruniu
```
![PolDeepNer interactive test demo](docs/media/PolDeepNer_demo.gif)


To exit press Enter without typing any text.

#### Process a single plain text file

Sample file: [KPWr: Toronto Dominion Center](poldeepner/data/kpwr-toronto.txt)

Run:

```bash
python poldeepner/core/process_file.py -m nkjp-poleval18 -i poldeepner/data/kpwr-toronto.txt
```

Expected output:

```bash
[  0: 16] orgName              Toronto Dominion
[  0:  7] placeName_settlement Toronto
[ 85: 96] placeName_country    kanadyjskim
[105:112] placeName_settlement Toronto
[116:134] orgName              Financial District
[204:229] persName             Ludwiga Mies van der Rohe
[204:211] persName_forename    Ludwiga
[212:216] persName_surname     Mies
[225:229] persName_surname     Rohe
[236:243] orgName              Budynki
[293:303] geogName             Joe Fafard
[503:550] geogName             Inuitów – Toronto Dominion Gallery of Inuit Art
[513:520] placeName_settlement Toronto
```

Run:

```bash
python poldeepner/core/process_file.py -m n82-ft-kgr10 -i poldeepner/data/kpwr-toronto.txt
```

Expected output:

```bash
[  0: 23] nam_fac_goe          Toronto Dominion Centre
[ 28: 51] nam_fac_goe          Toronto Dominion Centre
[ 85: 96] nam_adj_country      kanadyjskim
[105:112] nam_loc_gpe_city     Toronto
[116:134] nam_loc_gpe_district Financial District
[204:229] nam_liv_person       Ludwiga Mies van der Rohe
[293:303] nam_liv_person       Joe Fafard
[503:510] nam_org_nation       Inuitów
[513:550] nam_fac_goe          Toronto Dominion Gallery of Inuit Art
```


### Training

Train each model separately. The training will replace the pre-trained models. 
```bash
python poldeepner/core/trainmodel.py \
              -i poldeepner/data/nkjp-nested-simplified-v2.iob \
              -t poldeepner/data/nkjp-nested-simplified-v2.iob \
              -f ft:poldeepner/model/cc.pl.300.bin \
              -m poldeepner/model/poldeepner-nkjp-ftcc-bigru \
              -e 15 -n GRU
                          
python poldeepner/core/trainmodel.py \
              -i poldeepner/data/nkjp-nested-simplified-v2.iob \
              -t poldeepner/data/nkjp-nested-simplified-v2.iob \
              -f ft:poldeepner/model/kgr10_orths.vec.bin \
              -m poldeepner/model/poldeepner-nkjp-ftkgr10orth-bigru \
              -e 15 -n GRU
                          
python poldeepner/core/trainmodel.py \
              -i poldeepner/data/nkjp-nested-simplified-v2.iob \
              -t poldeepner/data/nkjp-nested-simplified-v2.iob \
              -f ft:poldeepner/model/kgr10-plain-sg-300-mC50.bin \
              -m poldeepner/model/poldeepner-nkjp-ftkgr10plain-lstm \
              -e 15 -n LSTM
```

Before training the structure of the NN will be printed:
```bash
Layer (type)                 Output Shape              Param #   
=================================================================
word_input (InputLayer)      (None, None, 100)         0         
_________________________________________________________________
dropout_1 (Dropout)          (None, None, 100)         0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, None, 200)         120600    
_________________________________________________________________
dense_1 (Dense)              (None, None, 100)         20100     
_________________________________________________________________
crf_1 (CRF)                  (None, None, 41)          5904      
=================================================================
Total params: 146,604
Trainable params: 146,604
Non-trainable params: 0
_________________________________________________________________
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


KPWr (n82 NER model)
-----------------------------------------------------------

![KPWr n82 evaluation](docs/media/kpwr-n82-pprai.png)

### Evaluate

```bash
python poldeepner/core/eval.py \
  --model MODEL \
  --input poldeepner/data/kpwr-ner-n82-test.iob
```

Replace MODEL with value from the Model column from the table below.

| Model                    | Precision | Recall | F1    |    Time | Memory usage (peak) [1] | Embeddings size |
|--------------------------|----------:|-------:|------:|--------:|------------------------:|----------------:|  
| n82-elmo-kgr10 [2]       |     73.97 |  75.49 | 74.72 |  <  5 m |                  4.5 GB |          0.4 GB |
| n82-pprai                |     74.70 |  71.92 | 73.28 |  < 12 m |          41.5 (54.0) GB |         38.1 GB |
| n82-ft-kgr10             |     71.36 |  71.49 | 71.43 |  <  2 m |          13.5 (22.5) GB |         10.8 GB |
| n82-ft-ccmaca            |     72.74 |  68.89 | 70.76 |  <  7 m |          24.1 (42.0) GB |         20.1 GB |
| n82-ft-cc                |     71.40 |  66.43 | 68.83 |  <  3 m |           9.2 (14.8) GB |          7.2 GB |
| n82-ft-kgr10-compact [2] |     69.51 |  67.04 | 68.25 |  <  2 m |           7.0  (8.8) GB |          3.7 GB |
| Liner2 [3]               |     67.65 |  58.85 | 62.93 |  <  8 m |           3.0 GB        |          0.5 GB |

[1] Memory usage was measured using [psrecord](https://pypi.org/project/psrecord/):

```bash
psrecord PID --interval 1 --plot plot.png --log activity.txt
```

[2] The results are approximate and they will be updated when the model is published.

[3] [Liner2](https://github.com/CLARIN-PL/Liner2) is using Conditional Random Fields.




Train and test a custom model (example for n82)
-----------------------------------------------------------

### Train

```bash
python poldeepner/core/trainmodel.py \
              -i poldeepner/data/kpwr-ner-n82-train-tune.iob \
              -t poldeepner/data/kpwr-ner-n82-test.iob \
              -f ft:poldeepner/model/kgr10.plain.skipgram.dim300.neg10.bin \
              -m poldeepner/tmp/poldeepner-n82-ftkgr10-bigru \
              -e 30 -n GRU
```

### Evaluate

```bash
python poldeepner/core/eval.py \
  --model poldeepner/tmp/poldeepner-n82-ftkgr10-bigru \
  --embeddings ft:poldeepner/model/kgr10.plain.skipgram.dim300.neg10.bin \
  --input poldeepner/data/kpwr-ner-n82-test.iob
```

```bash
annotation                  TP      FP      FN precision    recall  f1-score   support
(...)
TOTAL                     3104    1160    1326     72.80     70.07     71.41      4430
```
Processing time: 2m; 
Memory usage: 14GB; 
Tokens: ~80k

License
-----------------------------------------------------------
Copyright (C) Wrocław University of Science and Technology (PWr), 2010-2020. All rights reserved.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.