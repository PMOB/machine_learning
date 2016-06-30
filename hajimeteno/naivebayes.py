#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import log


class NaiveBayes:
    def __init__(self, extractor):
        self.words = dict()
        self.categories = dict()
        self.extractor = extractor

    def inc_category(self, cat):
        self.categories.setdefault(cat, 1)
        self.categories[cat] += 1

    def inc_word(self, word, cat):
        self.words.setdefault(cat, dict())
        self.words[cat].setdefault(word, 1)
        self.words[cat][word] += 1

    def count_category(self, cat):
        if cat in self.categories:
            return self.categories[cat]
        return 1

    def count_word(self, word, cat):
        if cat in self.words:
            if word in self.words[cat]:
                return self.words[cat][word]
        return 1

    def train(self, doc, cat):
        words = self.extractor(doc)
        for word in words:
            self.inc_word(word, cat)
        self.inc_category(cat)

    def classify(self, doc):
        words = self.extractor(doc)
        posterior = lambda cat: self.posterior(words, cat)
        scores = map(posterior, self.categories.keys())
        _, cat = max(scores)
        return cat

    def posterior(self, words, cat):
        total = sum(self.categories.values())
        prior = log(self.categories[cat]/total)
        around_cat = lambda word: self.count_word(word, cat)
        evidence = sum(map(around_cat, words))
        bayes = lambda w: log(self.count_word(w, cat) / evidence)
        likelihood = sum(map(bayes, words))
        return prior + likelihood, cat
