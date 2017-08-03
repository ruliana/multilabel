# -*- coding: utf-8 -*-
# Remove records and labels that are less representative
import re
import json
import codecs
from collections import Counter
INPUT_FILE = 'vagas_e_cargos.json'
OUTPUT_FILE = 'vagas_e_cargos_01.json'

# Minimum people on a job
MIN_PEOPLE = 50
# Mininum percent of people in a label (job description)
MIN_PERC_LABEL = 0.05

def removeHTML(text):
    "Remove HTML tags and entities from text"
    text = re.sub('<[^>]+?>|&[^;]+;', ' ', text)
    text = re.sub('\r?\n', ' ', text)
    text = re.sub('\\s+', ' ', text)
    return text

qtde_pre = 0
counter_pre = Counter()
unique_pre = set()

qtde_pos = 0
counter_pos = Counter()
unique_pos = set()

with codecs.open(INPUT_FILE, 'r', encoding='utf-8') as file_in:
    with codecs.open(OUTPUT_FILE, 'w', encoding='utf-8') as file_out:
        for line in file_in:
            record = json.loads(line)
            if not record.has_key(u'cargos'): continue

            identifier = record[u'Cod_vaga']
            title = record[u'cargo_vaga']
            description = removeHTML(record[u'AnuncioWeb_vaga'])
            labels = [x for x in record[u'cargos'] if x[0] != None and x[0].strip() != '']

            qtde_pre += 1
            #if qtde_pre > 10: break
            lbls = frozenset([text for text, number in labels])
            counter_pre[lbls] += 1
            unique_pre.update(lbls)

            labels_sum = float(sum([x[1] for x in labels]))
            if labels_sum < MIN_PEOPLE: continue

            labels = [x for x in labels if x[1] / labels_sum > MIN_PERC_LABEL]
            if len(labels) == 0: continue

            qtde_pos += 1
            lbls = frozenset([text for text, number in labels])
            counter_pos[lbls] += 1
            unique_pos.update(lbls)

            file_out.write(json.dumps([identifier, title, description, labels]) + '\n')

print('Antes da limpeza')
ulabl_pre = len(unique_pre)
label_pre = sum([len(s) * q for s, q in counter_pre.most_common()])
diver_pre = len(counter_pre)
divnr_pre = diver_pre / float(qtde_pre)
lcard_pre = label_pre / float(qtde_pre)
ldens_pre = lcard_pre / float(qtde_pre)
print('    Registros: %d' % qtde_pre)
print('      Rótulos: %d' % ulabl_pre)
print('  Diversidade: %d' % diver_pre)
print('Divers. Norm.: %0.2f' % divnr_pre)
print('Cardinalidade: %0.2f' % lcard_pre)
print('    Densidade: %0.8f' % ldens_pre)

print('Após a limpeza')
ulabl_pos = len(unique_pos)
label_pos = sum([len(s) * q for s, q in counter_pos.most_common()])
diver_pos = len(counter_pos)
divnr_pos = diver_pos / float(qtde_pos)
lcard_pos = label_pos / float(qtde_pos)
ldens_pos = lcard_pos / float(qtde_pos)
print('    Registros: %d' % qtde_pos)
print('      Rótulos: %d' % ulabl_pos)
print('  Diversidade: %d' % diver_pos)
print('Divers. Norm.: %0.2f' % divnr_pos)
print('Cardinalidade: %0.2f' % lcard_pos)
print('    Densidade: %0.8f' % ldens_pos)
