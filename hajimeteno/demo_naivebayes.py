#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import urllib.request
from bs4 import BeautifulSoup as bs
from naivebayes import NaiveBayes


extractor = re.compile('\w+')
def get_words(doc):
    words = extractor.findall(doc)
    return [w.lower() for w in words]


def readdoc(soup):
    for tag in soup.find_all('p'):
        for sentence in tag.text.split('.'):
            yield sentence + '.'


def fetch(url, fname):
    with open(fname, 'w') as f:
        src = urllib.request.urlopen(url)
        soup = bs(src.read(), "html.parser")
        f.writelines(readdoc(soup))


def fetch_all():
    fetch("http://animals.mom.me/countries-pythons-live-2498.html", "snakes1")
    fetch("http://www.kidzone.ws/lw/snakes/facts-python.htm", "snakes2")
    fetch("https://docs.python.org/3/tutorial/index.html",
            "ProgrammingLanguage1")
    fetch("https://docs.python.org/3/tutorial/appetite.html",
            "ProgrammingLanguage2")


def train(classifier):
    dataset = [
            ("ProgrammingLanguage1", "Programming Language"),
            ("ProgrammingLanguage2", "Programming Language"),
            ("snakes1", "snake"),
            ("snakes2", "snake"),
            ]
    for fname, cat in dataset:
        f = open(fname)
        for line in f.readlines():
            classifier.train(line, cat)
        f.close()
    return classifier


if __name__ == '__main__':
    # fetch_all()
    classifier = NaiveBayes(get_words)
    learned = train(classifier)
    line = input('> ')
    cat = learned.classify(line)
    print('You ment %s.' % cat)
