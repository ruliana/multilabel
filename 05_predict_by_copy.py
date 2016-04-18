# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
from random import sample
from load_data import load_data_raw
from tokenizer import tokenize
from collections import Counter
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, clone

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from sklearn.cross_validation import train_test_split, StratifiedKFold

from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier

from unbalanced_dataset import UnderSampler, NearMiss, CondensedNearestNeighbour, OneSidedSelection,\
NeighbourhoodCleaningRule, TomekLinks, ClusterCentroids, OverSampler, SMOTE,\
SMOTETomek, SMOTEENN, EasyEnsemble, BalanceCascade

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def tfidf_vectorizer():
    return TfidfVectorizer(sublinear_tf=True,
                           norm='l2',
                           analyzer='word',
                           strip_accents=None,
                           stop_words=None,
                           # A tokenização tb faz stemming e remove acentos
                           tokenizer=tokenize,
                           min_df=1,
                           max_df=0.5)

def count_vectorizer():
    return CountVectorizer(analyzer='word',
                           strip_accents=None,
                           stop_words=None,
                           # A tokenização tb faz stemming e remove acentos
                           tokenizer=tokenize,
                           min_df=1)

class MultiLabelEmsemble(BaseEstimator):
    def __init__(self, clf=KNeighborsClassifier(4, weight='uniform')):
        self.clf = clf
        self.clf_bag = []

    def fit(self, X, y):
        everything = frozenset(X)
        y = np.array(y)
        table = np.c_[X, y]

        y_unique = Counter(map(frozenset, y))
        for y_target, _ in y_unique.most_common():
            good = frozenset(table[y == y_target, 0])
            bad = everything - good

            good = list(good)
            bad = list(bad)

            good_and_bad = good + bad
            labels = ([1] * len(good)) + ([0] * len(bad))
            classifier = clone(self.clf)
            self.clf_bag.append((y_target, classifier.fit(good_and_bad, labels)))

        return self

    def predict(self, X):
        rslts = [Counter() for _ in X]
        for target, clf in self.clf_bag:
            ps = clf.predict(X)
            for i, p in enumerate(ps):
                if p == 1:
                    rslts[i].update(target)
        return rslts

full_set = load_data_raw('data/amostra_10perc_6.json')

X = np.array([title for identity, title, text, label in full_set])
y = np.array([frozenset(label) for identity, title, text, label in full_set])

targets = [u'civil engenheiro', u'vendedor', u'enfermeiro', u'arquiteto']
fs = [frozenset([i for i, labels in enumerate(y) if t in labels]) for t in targets]
z = list(reduce(lambda x, y: x | y, fs))
X = X[z]
y = y[z]

clfs = []
for n in range(10):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    pipeline = Pipeline(steps=[('vectorizer', tfidf_vectorizer()),
                               ('classifier', SGDClassifier(loss='hinge', class_weight='balanced'))])

    clf = MultiLabelEmsemble(clf=pipeline)
    clf.fit(x_train, y_train)
    clfs.append(clf)

preds = [clf.predict(x_test) for clf in clfs]
preds = [reduce(lambda x, y: x + y, ps) for ps in zip(*preds)]

for n in range(10, 20):
    print('=' * 60)
    print('Threshold: %d' % n)

    preds1 = [frozenset([t for t, c in p.most_common() if c > n]) for p in preds]

    precs = []
    recs = []
    f1s = []
    for sample, real, predicted in zip(x_test, y_test, preds1):
        real = frozenset(real)
        tp = float(len(real & predicted))
        prec = tp / len(predicted) if len(predicted) > 0 else 0
        rec = tp / len(real) if len(real) > 0 else 0
        f1 = 2 * ((prec * rec) / (prec + rec)) if prec + rec > 0 else 0
        real_str = '[%s]' % ', '.join(real)
        predicted_str =  '[%s]' % ', '.join(predicted)
        # print('[prec: %0.2f, rec: %0.2f, f1: %0.2f] %s => %s | %s' % (prec, rec, f1, sample, real_str, predicted_str))
        print('%0.3f\t%0.3f\t%0.3f' % (prec, rec, f1))
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    print('-' * 60)
    print('Precisão: %0.2f' % (sum(precs) / len(precs)))
    print('1ºQ Precisão: %0.2f' % (np.percentile(precs, 25)))
    print('Med Precisão: %0.2f' % (np.median(precs)))
    print('3ºQ Precisão: %0.2f' % (np.percentile(precs, 75)))

    print('-' * 60)
    print('Abrangência: %0.2f' % (sum(recs) / len(recs)))
    print('1ºQ Abrangência: %0.2f' % (np.percentile(recs, 25)))
    print('Med Abrangência: %0.2f' % (np.median(recs)))
    print('3ºQ Abrangência: %0.2f' % (np.percentile(recs, 75)))

    print('-' * 60)
    print('F1 Score: %0.2f' % (sum(f1s) / len(f1s)))
    print('1ºQ F1 Score: %0.2f' % (np.percentile(f1s, 25)))
    print('Med F1 Score: %0.2f' % (np.median(f1s)))
    print('3ºQ F1 Score: %0.2f' % (np.percentile(f1s, 75)))

