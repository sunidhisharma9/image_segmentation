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
    def make_n_fold_test(data_vectorizer, n):
        X = data_vectorizer.X
        Y = data_vectorizer.Y
        classes = data_vectorizer.classes
        num_classes = len(classes)
        num_elements = np.shape(X)[0]
        num_elements_for_class = (num_elements/n)/num_classes
        
        X_for_class = []
        X_index = [(index, x) for (index, x) in enumerate(X)]

        size_view_shape = 9
        size_view_rgb = 10
        
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
        accuracies_bayes_shape = []
        accuracies_bayes_rgb = []
        accuracies_knn_shape = []
        accuracies_knn_rgb = []
        accuracies_majority = []
        
        for k in xrange(n):
            print 'Iteration', k , 'of', n
            test = folds[k]
            train = []
            for i in xrange(n):
                if i != k:
                    train += folds[i]

            X_train = np.array([t[0] for t in train])
            Y_train = np.array([t[1] for t in train]).ravel()
            X_test = np.array([t[0] for t in test])
            Y_test = np.array([t[1] for t in test]).ravel()

            len_train = np.shape(X_train)[0]
            len_test = np.shape(X_test)[0]
            
            shape_mapping_train = np.ix_(range(len_train), range(size_view_shape))
            shape_mapping_test = np.ix_(range(len_test), range(size_view_shape))
            rgb_mapping_train = np.ix_(range(len_train), range(size_view_shape, size_view_shape + size_view_rgb))
            rgb_mapping_test = np.ix_(range(len_test), range(size_view_shape, size_view_shape + size_view_rgb)) 

            X_train_shape = X_train[shape_mapping_train]
            X_test_shape = X_test[shape_mapping_test]
            X_train_rgb = X_train[rgb_mapping_train]
            X_test_rgb = X_test[rgb_mapping_test]
            
            classifier_bayes_shape = BayesClassifier(X_train_shape, Y_train, classes)
            classifier_bayes_rgb = BayesClassifier(X_train_rgb, Y_train, classes)
            classifier_knn_shape = KnnClassifier(X_train_shape, Y_train, 7)
            classifier_knn_rgb = KnnClassifier(X_train_rgb, Y_train, 7)
            classifier_majority = MajorityVoteClassifier(X_train, Y_train, classes)

            accuracies_bayes_shape.append(classifier_bayes_shape.evaluate(X_test_shape, Y_test))
            accuracies_bayes_rgb.append(classifier_bayes_rgb.evaluate(X_test_rgb, Y_test))
            accuracies_knn_shape.append(classifier_knn_shape.evaluate(X_test_shape, Y_test))
            accuracies_knn_rgb.append(classifier_knn_rgb.evaluate(X_test_rgb, Y_test))
            accuracies_majority.append(classifier_majority.evaluate(X_test, Y_test))

        accuracies = {'accuracies_bayes_shape': accuracies_bayes_shape,
                      'accuracies_bayes_rgb': accuracies_bayes_rgb,
                      'accuracies_knn_shape': accuracies_knn_shape,
                      'accuracies_knn_rgb': accuracies_knn_rgb,
                      'accuracies_majority': accuracies_majority}
        
        return accuracies
