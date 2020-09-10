def load_data_and_labels(filename, extra_features=False):
    sents, labels, dockstarts = [], [], []
    words, tags, dockstart = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip()
            if "DOCSTART" in line:
                dockstart.append(line)
                continue
            if line:
                cols = line.split('\t')
                if extra_features:
                    words.append([cols[0]] + cols[3:-1])
                else:
                    words.append(cols[0])
                tags.append(cols[-1])
            elif len(words) > 0:
                sents.append(words)
                labels.append(tags)
                dockstarts.append(dockstart)
                words, tags, dockstart = [], [], []
        return sents, labels, dockstarts