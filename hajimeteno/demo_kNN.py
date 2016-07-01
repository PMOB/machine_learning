#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import random
import urllib.request
from kNN import kNN


def fetch(url, fname):
    with open(fname, 'w') as f:
        src = urllib.request.urlopen(url)
        data = src.read()
        f.write(data.decode('utf-8'))


def fetch_all():
    fetch("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
            "iris.data")


def get_dataset():
    with open('iris.data') as csvfile:
        reader = csv.reader(csvfile)
        return list(reader)[:-1]


def train(classifier, dataset):
    for feature, label in dataset:
        classifier.train(feature, label)

    return classifier

def form(dataset):
    for record in dataset:
        feature = map(float, record[:-1])
        label = record[-1] 
        yield tuple(feature), label


def main():
    classifier = kNN()

    dataset = get_dataset()
    dataset = list(form(dataset))
    random.shuffle(dataset)
    training = dataset[:100]
    testing = dataset[100:]

    classifier = train(classifier, training)
    before = classifier.size
    classifier.compress(3.0)
    after = classifier.size

    correct = 0
    for feature, label in testing:
        seems = classifier.classify(feature)
        if seems == label:
            correct += 1

    print(correct/len(testing), before, after)


if __name__ == '__main__':
    # fetch_all()
    for _ in range(30):
        main()
