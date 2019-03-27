import os
import xml.etree.ElementTree as ET


def load_data(data_file_path):
    ''' Function for loading data from .iob, .xml (CCL format), .tok files or file with indices to such files
    :param data_file_path: path to iob, xml or index file with data
    '''

    x_data, y_data = [], []

    # Get data from iob file
    if data_file_path.endswith('.iob'):
        x_data, y_data, ext_data = load_iob(data_file_path, extended_data=True)
        return x_data, y_data, ext_data

    # Get data from xml file
    elif data_file_path.endswith('.xml'):
        x_data, y_data = load_xml(data_file_path)

    # Get data from index file
    else:
        with open(data_file_path, 'r') as index_file:
            for index in index_file:
                index = index.replace('\n', '')
                file_path = os.path.join(os.path.dirname(data_file_path), index)
                # Get data from iob listed in index file
                if index.endswith('.iob'):
                    x, y = load_iob(file_path)
                    x_data += x
                    y_data += y

                # Get data from xml listed in index file
                elif index.endswith('xml'):
                    x, y = load_xml(file_path)
                    x_data += x
                    y_data += y

                else:
                    raise UnsupportedFileFormat('Unsupported file format of file: ' + os.path.basename(index))
    return x_data, y_data


def load_iob(filename, extended_data=False):
    """Loads data and label from a file.

    Args:
        filename (str): path to the file.
        extended_data (bool): return columns other then tokens and annotations from iob

        The file format is tab-separated values.
        A blank line is required at the end of a sentence.

        For example:
        ```
        EU	B-ORG
        rejects	O
        German	B-MISC
        call	O
        to	O
        boycott	O
        British	B-MISC
        lamb	O
        .	O

        Peter	B-PER
        Blackburn	I-PER
        ...
        ```

    Returns:
        tuple(numpy array, numpy array): data and labels.

    Example:
        >>> filename = 'conll2003/en/ner/train.txt'
        >>> data, labels = load_data_and_labels(filename)
    """
    sents, labels = [], []
    words, tags = [], []
    ext_data_sents, ext_data = [], []
    with open(filename, 'r') as f:
        for line in f:
            if "DOCSTART" in line:
                continue
            line = line.rstrip()
            if line:
                cols = line.split('\t')
                words.append(cols[0])
                tags.append(cols[-1])
                if extended_data:
                    ext_data.append(cols[1:-1])
            else:
                sents.append(words)
                ext_data_sents.append(ext_data)
                labels.append(tags)
                words, tags, ext_data = [], [], []
        return sents, labels, ext_data_sents


class AnnsChannels:
    def __init__(self, anns):
        self.dict = {ann.get('chan'): 0 for ann in anns}

    def get_channel(self, ann_name):
        return self.dict[ann_name]

    def set_channel(self, ann_name, channel_value):
        self.dict[ann_name] = channel_value


def load_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    sentences = []
    labels = []
    for chunk in root.findall('chunk'):
        for sent in chunk.findall('sentence'):
            sentence = []
            anns_channels = AnnsChannels(sent.find('tok').findall('ann'))
            label = []
            for token in sent.findall('tok'):
                anns = token.findall('ann')
                annotation = ""
                if anns:
                    for ann in anns:
                        channel_value = int(ann.text)
                        annotation_name = ann.get('chan')
                        if channel_value > 0:
                            if annotation != "":
                                annotation += "#"
                            if anns_channels.get_channel(annotation_name) == channel_value:
                                annotation += "I-" + annotation_name
                            else:
                                annotation += "B-" + annotation_name
                            anns_channels.set_channel(annotation_name, channel_value)
                        else:
                            anns_channels.set_channel(annotation_name, channel_value)
                    if annotation == "":
                        annotation = "O"
                sentence.append(token.find('orth').text)
                label.append(annotation)
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels


def load_toki(file_path):
    with open(file_path) as f:
        data = []
        sentence = []
        for line in f:
            line_splitted = line.split('\t')
            if line_splitted[1] == 'newline':
                data.append(sentence)
                sentence = []
                continue
            else:
                sentence.append(line_splitted[0])
    return data


class UnsupportedFileFormat(Exception):
    def __init__(self, message):
        super(UnsupportedFileFormat, self).__init__(message)
