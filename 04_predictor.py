# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from tokenizer import tokenize
from load_data import load_data
from collections import Counter
from scipy.sparse import vstack
from random import sample

class ImbalancedClassifier(BaseEstimator):
    def undersampled(self, x, y):
        a_set = zip(x, y)
        corr = [elem for elem in a_set if elem[1] == 1]
        if len(corr) == 0:
            raise StandardError('Amostra de tamanho %d tem zero exemplos positivos' % len(a_set))
        miss = [elem for elem in a_set if elem[1] == 0]
        miss = sample(miss, len(corr))
        rslt = corr + miss
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

# tfidf = joblib.load('models/tf-idf.pk1')
# print('TF-IDF carregado')

full_set = load_data('data/amostra_10perc_2.json')
labels = [label for title, text, label in full_set]

distinct_labels = Counter(labels)
print('Rótulos distintos: %d' % len(distinct_labels))

for target, size in distinct_labels.most_common():
    if size < 10:
        break
    all_texts = frozenset([title for title, text, label in full_set])
    good_texts = frozenset([title for title, text, label in full_set if label == target])
    bad_texts = all_texts - good_texts

    print(u'Rótulo "%s"' % target)
    print(u' rótulo = %d' % len(good_texts))
    print(u'¬rótulo = %d' % len(bad_texts))

    intersection = frozenset.intersection(good_texts, bad_texts)
    if len(intersection) > 0:
        raise StandardError('Textos iguais aparecem em ambos os conjuntos: %d' % len(intersection))

    x = list(good_texts) + list(bad_texts)
    y = ([1] * len(good_texts)) + ([0] * len(bad_texts))

    # print('\n'.join(sample(good_texts, 50)))
    # print('=' * 50)
    # print('\n'.join(sample(bad_texts, 50)))
    # exit()

    # params = [min_df for min_df in np.linspace(0.01, 0.1, 10)]
    # for min_df in params:
    #     print(u'Transformando, min_df: %0.2f' % min_df)

    # tfidf = TfidfVectorizer(sublinear_tf=True,
    #                         norm='l1',
    #                         analyzer='word',
    #                         strip_accents=None,
    #                         stop_words=stopwords.words('portuguese'),
    #                         # A tokenização tb faz stemming e remove acentos
    #                         tokenizer=tokenize
    #                         min_df=max(1, int(len(good_texts) * 0.01)),
    #                         max_df=0.5)
    # x = tfidf.fit_transform(list(good_texts) + list(bad_texts))
    # y = ([1] * len(good_texts)) + ([0] * len(bad_texts))
    # print(sample(tfidf.get_feature_names(), 100))
    # print('Exemplos: %d, Características %d' % x.shape)

    vectorizer = CountVectorizer(min_df=5,
                                 ngram_range=(1, 2),
                                 analyzer='word',
                                 strip_accents=None,
                                 stop_words=stopwords.words('portuguese'),
                                 # A tokenização tb faz stemming e remove acentos
                                 tokenizer=tokenize)
    x = vectorizer.fit_transform(x)
    #print(sample(vectorizer.get_feature_names(), 100))
    #print('Exemplos: %d, Características %d' % x.shape)

    clf = BernoulliNB()
    scores = cross_val_score(clf, x, y, cv=10, scoring='f1', n_jobs=3)
    scores_str = [('%0.5f' % x).replace('.', ',') for x in scores]
    print('Average: %0.2f' % (sum(scores) / len(scores)))

    # clf = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
    # scores = cross_val_score(clf, x, y, cv=10, scoring='f1')
    # scores = [('%0.5f' % x).replace('.', ',') for x in scores]
    # print('\t'.join(scores))

    # clf = SGDClassifier(loss='hinge', class_weight='balanced')
    # scores = cross_val_score(clf, x, y, cv=10, scoring='f1', n_jobs=3)
    # scores_str = [('%0.5f' % x).replace('.', ',') for x in scores]
    # print('\t'.join(scores_str))
    # print('Average: %0.2f' % (sum(scores) / len(scores)))

    # cv = ShuffleSplit(len(y), n_iter=3, test_size=0.2)

    # params = [alpha for alpha in np.linspace(0.1, 1, 5)]
    # for alpha in params:
    #     clf = ImbalancedClassifier(BernoulliNB(alpha=alpha), perc_voters=0.8, n_classifiers=20)
    #     scores = cross_val_score(clf, x, y, cv=cv, scoring='f1')
    #     print('alpha: %0.2f, score: %0.2f' % (alpha, (sum(scores) / len(scores))))
