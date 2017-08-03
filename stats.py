# -*- coding: utf-8 -*-
from load_data import load_data_raw

labels = frozenset([frozenset(label) for identity, title, text, label in load_data_raw('vagas_e_cargos_02.json')])
print(len(labels))
print(list(labels)[:20])

