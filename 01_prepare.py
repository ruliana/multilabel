# -*- coding: utf-8 -*-
# Remove records and labels that are less representative
import json
import codecs

INPUT_FILE = 'vagas_e_cargos.json'
OUTPUT_FILE = 'vagas_e_cargos_01.json'

# Minimum people on a job
MIN_PEOPLE = 50
# Mininum percent of people in a label (job description)
MIN_PERC_LABEL = 0.05

with codecs.open(INPUT_FILE, 'r', encoding='utf-8') as file_in:
    with codecs.open(OUTPUT_FILE, 'w', encoding='utf-8') as file_out:
        for line in file_in:
            record = json.loads(line)
            if not record.has_key(u'cargos'):
                continue

            title = record[u'cargo_vaga']
            text = record[u'AnuncioWeb_vaga']
            labels = [x for x in record[u'cargos'] if x[0] != None and x[0].strip() != '']

            labels_sum = float(sum([x[1] for x in labels]))
            if labels_sum < MIN_PEOPLE:
                continue

            labels = [x for x in labels if x[1] / labels_sum > MIN_PERC_LABEL]
            if len(labels) == 0:
                continue

            file_out.write(json.dumps([title, text, labels]) + '\n')
