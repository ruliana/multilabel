# -*- coding: utf-8 -*-
import json
import codecs
import numpy as np
from load_data import load_data
from random import shuffle
from sklearn.externals import joblib
from nltk.corpus import stopwords
from tokenizer import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT_FILE = 'vagas_e_cargos_02.json'

identities, texts, labels = zip(*load_data(INPUT_FILE))

tfidf = TfidfVectorizer(analyzer='word',
                        strip_accents=None,
                        stop_words=stopwords.words('portuguese'),
                        # A tokenização tb faz stemming
                        tokenizer=tokenize,
                        min_df=5)

tfidf.fit(texts)
joblib.dump(tfidf, 'models/tf-idf.pk1')
