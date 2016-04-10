# -*- coding: utf-8 -*-
import json
import codecs
import numpy as np
from load_data import load_data, data_vs_label
from random import shuffle
from sklearn.externals import joblib
from nltk.corpus import stopwords
from tokenizer import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT_FILE = 'vagas_e_cargos_02.json'

texts, labels = data_vs_label(load_data(INPUT_FILE))

tfidf = TfidfVectorizer(analyzer='word',
                        strip_accents=None,
                        stop_words=stopwords.words('portuguese'),
                        tokenizer=tokenize,
                        min_df=5)

tfidf.fit(texts)
joblib.dump(tfidf, 'models/tf-idf.pk1')
#tfidf = joblib.load('models/tf-idf.pk1')

texts = tfidf.transform(texts)
full_set = zip(texts, labels)
joblib.dump(full_set, 'data/tf-idf-data.pk1')
