# -*- coding: utf-8 -*-
import re
import unicodedata
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords

pattern = re.compile(r"\d")

def removeHTML(text):
    "Remove HTML tags and entities from text"
    return re.sub('<[^>]+?>|&[^;]+;', '', text)

def removeAccents(text):
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore')

def wordalize(text):
    "Break in words, avoid punctuation"
    words = re.findall(r'\w+', text, re.UNICODE)
    return [w.lower() for w in words]

aliases = {'jr': 'junior',
           'sr': 'senior',
           'pl': 'pleno',
           'i': 'junior',
           'ii': 'pleno',
           'iii': 'senior'}
def alias(word):
    return aliases.get(word, word)

stops = frozenset(stopwords.words('portuguese'))
def filterStopWords(words):
    return [w for w in words if w not in stops]

stemmer = RSLPStemmer()
def stemify(words):
    "Stem words"
    return [stemmer.stem(w) for w in words]

def tokenize(text):
    "Break in words, remove punctuation and stop words and steem every word"
    words = wordalize(removeHTML(text))
    words = [w for w in words if not pattern.match(w)]
    words = filterStopWords(words)
    words = map(alias, words)
    words = stemify(words)
    return [removeAccents(w) for w in words]

