# -*- coding: utf-8 -*-
import json
import codecs

def load_data(file_name):
    with codecs.open(file_name, 'r', encoding='utf-8') as file_in:
        return [line_to_record(line) for line in file_in]

def line_to_record(line):
    title, text, labels = json.loads(line)
    label = '|'.join(sorted(labels))
    return title + ' ' + text, label

def data_vs_label(a_set):
    texts = [text for (text, label) in a_set]
    labels = [label for (text, label) in a_set]
    return texts, labels
