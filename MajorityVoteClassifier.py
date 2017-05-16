import pandas as pd
import numpy as np
import math as m
import pickle
import warnings
import random

#ignore warnings
warnings.filterwarnings('ignore')

from classifiers.KnnClassifier import *
from classifiers.BayesClassifier import *
from classifiers.DataVectorizer import *

class MajorityVoteClassifier:
    def __init__(self, X_rgb, X_shape, Y, classes):
        self.X_rgb = X_rgb
        self.X_shape = X_shape
        self.classes = classes
        self.Y = Y

        self.bayes_rgb = BayesClassifier(self.X_rgb, self.Y, self.classes)
        self.bayes_shape = BayesClassifier(self.X_shape, self.Y, self.classes)
        self.knn_rgb = KnnClassifier(self.X_rgb, self.Y, 1)
        self.knn_shape = KnnClassifier(self.X_shape, self.Y, 1)

    def predict(self, x_rgb, x_shape):
        predict_b_rgb = self.bayes_rgb.predict(x_rgb)[0]
        predict_b_shape = self.bayes_shape.predict(x_shape)[0]
        predict_knn_rgb = self.knn_rgb.predict(x_rgb)[0]
        predict_knn_shape = self.knn_shape.predict(x_shape)[0]

        result_dic = {}
        for (index, c) in enumerate(self.classes):
            result_dic[index] = 0

        try:
            result_dic[predict_b_rgb] += 1
            result_dic[predict_b_shape] += 1
            result_dic[predict_knn_rgb] += 1
            result_dic[predict_knn_shape] += 1
        except:
            print 'Erro - result_dic', result_dic
            input('wait...')
        votes = sorted(result_dic.items(), key=operator.itemgetter(1), reverse=True)
        return votes[0]

    def evaluate(self, X_rgb, X_shape, Y):
        num_total = 0.0
        num_right = 0.0
        num_wrong = 0.0

        for (index, row) in enumerate(X_rgb):
            current_index = Y[index]
            row_rgb = X_rgb[index]
            row_shape = X_shape[index]
            
            predicted_index = self.predict(row_rgb, row_shape)[0]
            if predicted_index == current_index:
                num_right += 1.0
            else:
                num_wrong += 1.0
            num_total += 1.0

        return num_right/num_total

dv_rgb = DataVectorizer(filename='result_pickles/RGB_view.pickle')
dv_shape = DataVectorizer(filename='result_pickles/shape_view.pickle')

tam_x = np.shape(dv_rgb.X)[0]
all_data = []

print 'Creating Data set'
for i in xrange(tam_x):
    all_data.append([dv_rgb.X[i], dv_shape.X[i], dv_rgb.Y[i]])

random.shuffle(all_data)

len_all_data = len(all_data)
size_train = int(0.7*float(len_all_data))

train = all_data[:size_train]
test = all_data[size_train:]
X_rgb_train = np.array([x_rgb for (x_rgb, x_shape ,y) in train])
X_shape_train = np.array([x_shape for (x_rgb, x_shape, y) in train])
Y_train = np.array([y for (x_rgb, x_shape, y) in train])

X_rgb_test = np.array([x_rgb for (x_rgb, x_shape ,y) in test])
X_shape_test = np.array([x_shape for (x_rgb, x_shape ,y) in test])
Y_test = np.array([y for (x_rgb, x_shape ,y) in test])

mjv = MajorityVoteClassifier(X_rgb_train, X_shape_train, Y_train, dv_rgb.classes)
print 'Accuracy bayes rgb', mjv.bayes_rgb.evaluate(X_rgb_test, Y_test)
print 'Accuracy bayes shape', mjv.bayes_shape.evaluate(X_shape_test, Y_test)
print 'Accuracy knn rgb', mjv.knn_rgb.evaluate(X_rgb_test, Y_test)
print 'Accuracy knn shape', mjv.knn_shape.evaluate(X_shape_test, Y_test)
print 'Accuracy majority vote', mjv.evaluate(X_rgb_test, X_shape_test, Y_test)
