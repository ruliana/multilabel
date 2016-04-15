# -*- coding: utf-8 -*-
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
        rslts = [None] * len(X)
        for target, clf in self.clf_bag:
            ps = clf.predict(X)
            for i, p in enumerate(ps):
                if p == 1:
                    rslt = Counter() if rslts[i] == None else rslts[i]
                    rslts[i] = rslt
                    rslt.update(target)
        return rslts

full_set = load_data_raw('data/amostra_01perc_4.json')

X = np.array([title for identity, title, text, label in full_set])
y = np.array([frozenset(label) for identity, title, text, label in full_set])

targets = [u'estagio', u'vendedor']
fs = [frozenset([i for i, labels in enumerate(y) if t in labels]) for t in targets]
z = list(reduce(lambda x, y: x | y, fs))
X = X[z]
y = y[z]

#for text, label in zip(X[:30], y):
#    print('[%s] %s => %s' % (', '.join(label), text, tokenize(text)))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

pipeline = Pipeline(steps=[('vectorizer', tfidf_vectorizer()),
                           ('classifier', SGDClassifier(loss='hinge'))])

clf = MultiLabelEmsemble(clf=pipeline)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)

for i, predicted in enumerate(preds):
    sample = x_test[i]
    real = ', '.join(y_test[i])
    print('%s => %s | %s' % (sample, real, predicted))

