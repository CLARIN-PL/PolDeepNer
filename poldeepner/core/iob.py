def load_data_and_labels(filename, extra_features=False):
    sents, labels = [], []
    words, tags = [], []
    with open(filename, 'r') as f:
        for line in f:
            if "DOCSTART" in line:
                continue
            line = line.rstrip()
            if line:
                cols = line.split('\t')
                if extra_features:
                    words.append([cols[0]] + cols[3:-1])
                else:
                    words.append(cols[0])
                tags.append(cols[-1])
            else:
                sents.append(words)
                labels.append(tags)
                words, tags = [], []
        return sents, labels