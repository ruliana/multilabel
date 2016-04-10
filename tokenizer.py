# -*- coding: utf-8 -*-
import re
from nltk.stem import RSLPStemmer

def removeHTML(text):
    "Remove HTML tags and entities from text"
    return re.sub('<[^>]+?>|&[^;]+;', '', text)

def wordalize(text):
    "Break in words, avoid punctuation"
    return re.findall(r'\w+', text, re.UNICODE)

stemmer = RSLPStemmer()
def stemify(words):
    "Remove stop words and stem each word"
    return [stemmer.stem(word.lower()) for word in words]

def tokenize(text):
    "Break in words, remove punctuation and stop words and steam every word"
    return stemify(wordalize(removeHTML(text)))

