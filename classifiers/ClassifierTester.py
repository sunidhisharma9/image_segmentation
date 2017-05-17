from pandas import index

import pandas as pd
import numpy as np
import csv
import pickle
import warnings
import random

from classifiers.BayesClassifier import *
from classifiers.KnnClassifier import *
from classifiers.MajorityVoteClassifier import *

class ClassifierTester:

    @staticmethod
    def make_n_fold_test(data_vectorizer, n, type_classifier='bayes'):
        X = data_vectorizer.X
        Y = data_vectorizer.Y
        classes = data_vectorizer.classes
        num_classes = len(classes)
        num_elements = np.shape(X)[0]
        num_elements_for_class = (num_elements/n)/num_classes
        
        X_for_class = []
        X_index = [(index, x) for (index, x) in enumerate(X)]

        for i in xrange(num_classes):
            separated_X = [[x,i] for (index, x) in X_index if Y[index]==i]
            random.shuffle(separated_X)
            X_for_class.append(separated_X)

        folds = []

        for k in xrange(n):
            low_index = k*num_elements_for_class
            high_index = low_index + num_elements_for_class
            new_fold = []
            for i in xrange(num_classes):
                new_fold += X_for_class[i][low_index:high_index]
            random.shuffle(new_fold)
            folds.append(new_fold)

        print 'Folds created'
        accuracies = []
        for k in xrange(n):
            test = folds[k]
            train = []
            for i in xrange(n):
                if i != k:
                    train += folds[i]
            X_train = np.array([t[0] for t in train])
            Y_train = np.array([t[1] for t in train]).ravel()
            X_test = np.array([t[0] for t in test])
            Y_test = np.array([t[1] for t in test]).ravel()


            if type_classifier == 'bayes':
                classifier = BayesClassifier(X_train, Y_train, classes)
            elif type_classifier == 'knn':
                classifier = KnnClassifier(X_train, Y_train, 1)
            elif type_classifier == 'majority':
                classifier = MajorityVoteClassifier(X_train, Y_train, classes)
            else:
                raise ValueError('Unknown classifier: ' + type_classifier)
                
            new_accuracy = classifier.evaluate(X_test, Y_test)
            print 'Accuracy', k, 'of', n , new_accuracy
            accuracies.append(new_accuracy)
        return accuracies
