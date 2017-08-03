# -*- coding: utf-8 -*-
import json
import codecs

def load_data_raw(file_name):
    with codecs.open(file_name, 'r', encoding='utf-8') as file_in:
        return [json.loads(line) for line in file_in]

def load_data(file_name):
    with codecs.open(file_name, 'r', encoding='utf-8') as file_in:
        return [line_to_record(line) for line in file_in]

def line_to_record(line):
    title, text, labels = json.loads(line)
    label = '|'.join(sorted(labels))
    return title, text, label
