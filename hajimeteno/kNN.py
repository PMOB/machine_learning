#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from collections import Counter


def distance(v1, v2):
    dists = map(lambda a, b: (a-b)**2, v1, v2)
    return math.sqrt(sum(dists))


class Record:
    def __init__(self, feat, label, is_clue):
        self._feat = feat
        self._label = label
        self._is_clue = is_clue

    @property
    def is_clue(self):
        return self._is_clue

    @is_clue.setter
    def is_clue(self, value):
        self._is_clue = value

    def unpack(self):
        return self._feat, self._label, self._is_clue


class kNN:
    def __init__(self, k=5):
        self.database = list()
        self.k = k

    def train(self, feature, label):
        self.database.append(Record(feature, label, True))

    @property
    def size(self):
        clues = filter(lambda r: r.is_clue, self.database)
        return len(list(clues))

    def compress(self, e=0.1):
        for record in self.database:
            feat, label, _ = record.unpack()
            for another in self.database:
                another_feat, same, is_clue = another.unpack()
                if not is_clue or label == same:
                    continue
                if distance(feat, another_feat) < e:
                    record.is_clue = False
                else:
                    record.is_clue = True

    def classify(self, feature):
        def get_distances():
            for record in self.database:
                feat, label, is_clue = record.unpack()
                if is_clue:
                    yield distance(feature, feat), label

        distances = list(get_distances())
        nearest = sorted(distances)
        kneighbor = nearest[:self.k]
        count = Counter(map(lambda x: x[1], kneighbor))
        result = count.most_common()[0]
        return result[0]
