# -*- coding: utf-8 -*-
import json
import codecs
import itertools as itr
import yaml
from collections import Counter

INPUT_FILE = 'vagas_e_cargos_01.json'
OUTPUT_FILE = 'vagas_e_cargos_02.json'

# Minimum labelset repetition
CUT_PARAMETER = 100
# Minimum labelset size in rejected training set
WIDTH_PARAMETER = 2

def all_combinations(coll):
    combinations = set()
    for i in range(1, len(coll)):
        for combs in itr.combinations(coll, i):
            combinations.add(frozenset(combs))
    return combinations

# Phase 1 - Common subsets
label_counter = Counter()
trainingset = set()

# Read the training set, expected a cleaned one
with codecs.open(INPUT_FILE, 'r', encoding='utf-8') as file_in:
    for line in file_in:
        record = json.loads(line)

        identifier = record[0]
        title = record[1]
        text = record[2]
        labels = frozenset([x[0] for x in record[3]])

        label_counter[labels] += 1
        trainingset.add((identifier, title, text, labels))

labelset = frozenset([label for (label, count) in label_counter.iteritems() if count >= CUT_PARAMETER])
trainingset_first = frozenset([record for record in trainingset if record[3] in labelset])
trainingset_rejected = trainingset - trainingset_first

# Phase 2 - Pruned subsets
# All label subsets of the rejected records
# Keep only the ones already in the firt label subset
label_counter = Counter()
for (identifier, title, text, labels) in trainingset_rejected:
    new_labels = [label for label in all_combinations(labels) if len(label) > 0 and label in labelset]
    label_counter.update(new_labels)

# Keep only the labelsets width the same cut paramenter
# with at least some labels
labelset_second = frozenset([label for (label, count) in label_counter.iteritems() if count >= CUT_PARAMETER and len(label) >= WIDTH_PARAMETER])

# In the rejected training set, create a new record
# for each label subset found in the subset we just
# created
trainingset_second = set()
for (indentifier, title, text, labels) in trainingset_rejected:
    for label in all_combinations(labels):
        if label in labelset_second:
            trainingset_second.add((identifier, title, text, label))

with codecs.open(OUTPUT_FILE, 'w', encoding='utf-8') as file_out:
    for (identifier, title, text, labels) in trainingset_first.union(trainingset_second):
        file_out.write(json.dumps([identifier, title, text, list(labels)]) + '\n')
