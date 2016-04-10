# -*- coding: utf-8 -*-
import json
import codecs
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from load_data import load_data, data_vs_label
from collections import Counter
from scipy.sparse import vstack
from random import sample

class UnderSampler:
    def fit(x, y):
        a_set = zip(x, y)
        corr = [x for x in a_set if x[1] == 1]
        miss = [x for x in a_set if x[1] == 0]
        miss = sample(miss, len(corr))
        rslt = miss + corr
        return zip(*rlt)
    def transform(x, y):
        a_set = zip(x, y)
        corr = [x for x in a_set if x[1] == 1]
        miss = [x for x in a_set if x[1] == 0]
        miss = sample(miss, len(corr))
        rslt = miss + corr
        return zip(*rlt)

def train(a_set):
    the_set = under_sample(a_set)
    data, labels = data_vs_label(the_set)
    data = vstack(data)
    classifier = SGDClassifier(loss='hinge')
    scores = cross_val_score(classifier, data, labels, cv=10, scoring='f1', n_jobs=3)
    score = (sum(scores) / len(scores))
    return classifier, score

tfidf = joblib.load('models/tf-idf.pk1')
print('TF-IDF carregado')

full_set = joblib.load('data/sample-01perc.pk1')
texts, labels = data_vs_label(full_set)
print('Dados carregados')
print('textos/rótulos: %d/%d' % (len(texts), len(labels)))

distinct_labels = Counter(labels)
print('Rótulos distintos: %d' % len(distinct_labels))

for target, size in distinct_labels.most_common()[:1]:
    if size < 10:
        break
    print(u'Rótulo "%s": %d' % (target, size))
    a_set = [(text, (1 if label == target else 0)) for (text, label) in full_set]
    classifiers_and_scores = [train(a_set) for _ in range(1, 11)]
    classifiers = [clf for clf, score in classifiers_and_scores]
    scores = [score for clf, score in classifiers_and_scores]
    score = (sum(scores) / len(scores))
    print(score)


