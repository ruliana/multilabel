# -*- coding: utf-8 -*-
from load_data import load_data
from math import floor
from random import sample
from sklearn.externals import joblib

data = joblib.load('data/tf-idf-data.pk1')
part10 = sample(data, int(floor(len(data) * 0.1)))
part01 = sample(data, int(floor(len(data) * 0.01)))
joblib.dump(part10, 'data/sample-10perc.pk1')
joblib.dump(part01, 'data/sample-01perc.pk1')

