import operator
import random
import numpy as np
import pandas as pd
import pickle

from classifiers.DataVectorizer import *
from classifiers.KnnClassifier import *

print 'Testing RGB view'
dv = DataVectorizer(filename='result_pickles/RGB_view.pickle')
tam_x = np.shape(dv.X)[0]
all_data = []

print 'Creating Data set'
for i in xrange(tam_x):
    all_data.append([dv.X[i], dv.Y[i]])

random.shuffle(all_data)

len_all_data = len(all_data)
size_train = int(0.7*float(len_all_data))
size_validation = int(0.15*float(len_all_data))
size_test = int(0.15*float(len_all_data))

train = all_data[:size_train]
validation = all_data[size_train+1:size_train + size_validation]
X_train = np.array([x for (x,y) in train])
Y_train = np.array([y for (x,y) in train])
X_validation = np.array([x for (x,y) in validation])
Y_validation = np.array([y for (x,y) in validation])

accuracies = []
for i in xrange(49):
    print 'Iteration', i, 'of 50'
    K = 2*i + 1
    knn = KnnClassifier(X_train, Y_train, K)
    new_accuracy = knn.evaluate(X_validation, Y_validation)
    print 'New Accuracy', new_accuracy
    accuracies.append([K, new_accuracy])

out = open('result_pickles/knn_accuracies_for_k.pickle', 'wb')
pickle.dump(accuracies, out)
out.close()