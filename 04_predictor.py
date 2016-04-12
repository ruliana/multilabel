# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from load_data import load_data, data_vs_label
from collections import Counter
from scipy.sparse import vstack
from random import sample

class ImbalancedClassifier(BaseEstimator):
    def undersampled(self, x, y):
        a_set = zip(x, y)
        corr = [elem for elem in a_set if elem[1] == 1]
        miss = [elem for elem in a_set if elem[1] == 0]
        miss = sample(miss, len(corr))
        rslt = miss + corr
        nx, ny = zip(*rslt)
        return vstack(nx), ny

    def __init__(self, classifier, n_classifiers=50, perc_voters=0.5, sampler=None):
        self.classifiers = []
        self.classifier = classifier
        self.n_classifiers = n_classifiers
        self.perc_voters = perc_voters
        self.sampler = self.undersampled if sampler == None else sampler

    def fit(self, x, y):
        self.classifiers = [self.fit_one(x, y) for _ in xrange(self.n_classifiers)]
        return self

    def fit_one(self, x, y):
        ux, uy = self.sampler(x, y)
        clf = clone(self.classifier)
        clf.fit(ux, uy)
        return clf

    def predict(self, x):
        def average(list_of_numbers):
            return sum(list_of_numbers) / float(len(list_of_numbers))
        partial_rslt = [clf.predict(x) for clf in self.classifiers]
        rslts = [average(preds) for preds in zip(*partial_rslt)]
        return [(1 if x >= self.perc_voters else 0) for x in rslts]

tfidf = joblib.load('models/tf-idf.pk1')
print('TF-IDF carregado')

full_set = joblib.load('data/sample-10perc.pk1')
texts, labels = data_vs_label(full_set)
print('Dados carregados')
print('textos/rótulos: %d/%d' % (len(texts), len(labels)))

distinct_labels = Counter(labels)
print('Rótulos distintos: %d' % len(distinct_labels))

for target, size in distinct_labels.most_common()[:1]:
    if size < 10:
        break
    print(u'Rótulo "%s": %d' % (target, size))
    a_set = [(text, (1 if label == target else 0)) for text, label in full_set]
    x, y = zip(*a_set)
    x = vstack(x)
    params = [(pv, nc) for pv in np.linspace(0.65, 0.95, 11)
                       for nc in [50]]
    for pv, nc in params:
        clf = ImbalancedClassifier(SGDClassifier(loss='hinge'), perc_voters=pv, n_classifiers=nc)
        scores = cross_val_score(clf, x, y, cv=10, scoring='f1')
        print('pv=%0.2f, %s' % (pv, scores))

#    classifiers_and_scores = [train(a_set) for _ in xrange(100)]
#    classifiers = [clf for clf, score in classifiers_and_scores]
#    scores = [score for clf, score in classifiers_and_scores]
#    score = average(scores)
#    print(score)
#    ens = BinaryEnsemble(classifiers, perc_voters=0.95)
#    data, trues = data_vs_label(a_set)
#    data = vstack(data)
#    preds = ens.predict(data)
#    print(confusion_matrix(trues, preds))
#    print(f1_score(trues, preds))


