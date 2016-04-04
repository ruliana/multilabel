# -*- coding: utf-8 -*-
import json
import codecs
from sklearn.feature_extraction.text import TfidfTransformer

INPUT_FILE = 'vagas_e_cargos_02.json'

trainingset = []
labelset = []
with codecs.open(INPUT_FILE, 'r', encoding='utf-8') as file_in:
    for line in file_in:
        record = json.loads(line)

        title = frozenset(record[0])
        text = frozenset(record[1])
        labels = frozenset(record[2])

        trainingset.append(title.union(text))
        labelset.append(labels)

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(training_set)

