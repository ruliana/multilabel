# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
from random import sample
from load_data import load_data_raw
from tokenizer import tokenize
from collections import Counter
from itertools import takewhile
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, clone
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
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

def fit_one(X, y, y_target, clf):
    good = frozenset([x for x in X[y == y_target]])
    bad  = frozenset([x for x in X[y != y_target]])
    bad = bad - good
    good_and_bad = list(good) + list(bad)
    labels = ([1] * len(good)) + ([0] * len(bad))
    classifier = clone(clf)
    return (y_target, classifier.fit(good_and_bad, labels))

class MultiLabelEmsemble(BaseEstimator):
    def __init__(self, clf=KNeighborsClassifier(4, weight='uniform'), n_jobs=1):
        self.clf = clf
        self.n_jobs = n_jobs
        self.clf_bag = []

    def fit(self, X, y):
        y_unique = frozenset(map(frozenset, y))
        if self.n_jobs == 1:
            self.clf_bag = [fit_one(X, y, y_target, self.clf) for y_target in y_unique]
        else:
            self.clf_bag = Parallel(n_jobs=self.n_jobs)(delayed(fit_one)(X, y, y_target, self.clf) for y_target in y_unique)
        return self

    def predict(self, X):
        rslts = [Counter() for _ in X]
        for target, clf in self.clf_bag:
            ps = clf.predict(X)
            for i, p in enumerate(ps):
                if p == 1:
                    rslts[i].update(target)
        return rslts

class BagEmsemble(BaseEstimator):
    def __init__(self, n_clf=10, train_size=0.67, clf=KNeighborsClassifier(4, weight='uniform')):
        self.clf = clf
        self.n_clf = n_clf
        self.train_size = train_size
        self.clf_bag = []

    def fit(self, X, y):
        for _ in range(self.n_clf):
            x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size)
            classifier = clone(self.clf)
            self.clf_bag.append(classifier.fit(x_train, y_train))
        return self

    def predict(self, X):
        preds = [clf.predict(X) for clf in self.clf_bag]
        preds = [reduce(lambda x, y: x + y, ps) for ps in zip(*preds)]
        return preds

    def select(self, counter):
        least = counter.most_common(3)[-1][1]
        rslt = takewhile(lambda (a, b): b >= least, counter.most_common())
        rslt = map(lambda (a, b): a, rslt)
        return frozenset(rslt)

class BagEnsembleSelector(BaseEstimator):
    def __init__(self, clf=MultiLabelEmsemble(), cut=3):
        self.clf = clf
        self.cut = cut

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        preds = self.clf.predict(X)
        return [self.select(rslt) for rslt in preds]

    def select(self, counter):
        if len(counter) == 0: return frozenset([])
        least = counter.most_common(self.cut)[-1][1]
        rslt = takewhile(lambda (a, b): b >= least, counter.most_common())
        rslt = map(lambda (a, b): a, rslt)
        return frozenset(rslt)

def load_sample(filename):
    full_set = load_data_raw(filename)

    X = np.array([title for identity, title, text, label in full_set])
    y = np.array([frozenset(label) for identity, title, text, label in full_set])

    # targets = [u'civil engenheiro', u'vendedor', u'enfermeiro', u'arquiteto', u'designer']
    # fs = [frozenset([i for i, labels in enumerate(y) if t in labels]) for t in targets]
    # z = list(reduce(lambda x, y: x | y, fs))
    # X = X[z]
    # y = y[z]
    return X, y

def validation(train_file, test_file):
    X, y = load_sample(train_file)

    sgd = SGDClassifier(loss='hinge', class_weight='balanced')
    pip = Pipeline(steps=[('vectorizer', tfidf_vectorizer()),
                          ('classifier', sgd)])
    mlt = MultiLabelEmsemble(clf=pip, n_jobs=3)
    # bag = BagEmsemble(clf=mlt, n_clf=10, train_size=0.67)
    sel = BagEnsembleSelector(clf=mlt, cut=2)

    classifier = sel
    classifier.fit(X, y)

    x_test, y_test = load_sample(test_file)
    preds = classifier.predict(x_test)

    precs = []
    recs = []
    f1s = []
    for sample, real, predicted in zip(x_test, y_test, preds):
        real = frozenset(real)
        tp = float(len(real & predicted))
        prec = tp / len(predicted) if len(predicted) > 0 else 0
        rec = tp / len(real) if len(real) > 0 else 0
        f1 = 2 * ((prec * rec) / (prec + rec)) if prec + rec > 0 else 0
        real_str = '[%s]' % ', '.join(real)
        predicted_str =  '[%s]' % ', '.join(predicted)
        print('[prec: %0.2f, rec: %0.2f, f1: %0.2f] %s => %s | %s' % (prec, rec, f1, sample, real_str, predicted_str))
        # print('%0.3f\t%0.3f\t%0.3f' % (prec, rec, f1))
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    print('-' * 60)
    print('Precisão: %0.2f' % (sum(precs) / len(precs)))
    print('Min Precisão: %0.2f' % (np.percentile(precs, 5)))
    print('1ºQ Precisão: %0.2f' % (np.percentile(precs, 25)))
    print('Med Precisão: %0.2f' % (np.median(precs)))
    print('3ºQ Precisão: %0.2f' % (np.percentile(precs, 75)))
    print('Max Precisão: %0.2f' % (np.percentile(precs, 95)))

    print('-' * 60)
    print('Revocação: %0.2f' % (sum(recs) / len(recs)))
    print('Min Revocação: %0.2f' % (np.percentile(recs, 5)))
    print('1ºQ Revocação: %0.2f' % (np.percentile(recs, 25)))
    print('Med Revocação: %0.2f' % (np.median(recs)))
    print('3ºQ Revocação: %0.2f' % (np.percentile(recs, 75)))
    print('Max Revocação: %0.2f' % (np.percentile(recs, 95)))

    print('-' * 60)
    print('F1 Score: %0.2f' % (sum(f1s) / len(f1s)))
    print('Min F1 Score: %0.2f' % (np.percentile(f1s, 5)))
    print('1ºQ F1 Score: %0.2f' % (np.percentile(f1s, 25)))
    print('Med F1 Score: %0.2f' % (np.median(f1s)))
    print('3ºQ F1 Score: %0.2f' % (np.percentile(f1s, 75)))
    print('Max F1 Score: %0.2f' % (np.percentile(f1s, 95)))

for i in range(0, 1):
    print('=' * 60)
    print('Treino/Teste %02d' % i)
    validation('data/train-%02d.json' % i, 'data/test-%02d.json' % i)
