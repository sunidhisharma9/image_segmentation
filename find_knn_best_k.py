import operator
import random
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from classifiers.DataVectorizer import *
from classifiers.KnnClassifier import *

print 'Testing RGB view'
dv = DataVectorizer(filename='result_pickles/full_view.pickle')
tam_x = np.shape(dv.X)[0]
all_data = []

print 'Creating Data set'
for i in xrange(tam_x):
    all_data.append([dv.X[i], dv.Y[i]])

accuracies = []

size_view_shape = 9
size_view_rgb = 10

for i in xrange(10):
    mean_accuracies = 0
    for j in xrange(10):
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

        len_train = np.shape(X_train)[0]
        len_validation = np.shape(X_validation)[0]
            
        shape_mapping_train = np.ix_(range(len_train), range(size_view_shape))
        shape_mapping_validation = np.ix_(range(len_validation), range(size_view_shape))
        rgb_mapping_train = np.ix_(range(len_train), range(size_view_shape, size_view_shape + size_view_rgb))
        rgb_mapping_validation = np.ix_(range(len_validation), range(size_view_shape, size_view_shape + size_view_rgb)) 

        print 'Iteration', i + 1, 'of 10'
        K = 2*i + 1
        knn_rgb = KnnClassifier(X_train[rgb_mapping_train], Y_train, K)
        knn_shape = KnnClassifier(X_train[shape_mapping_train], Y_train, K)
        new_accuracy_rgb = knn_rgb.evaluate(X_validation[rgb_mapping_validation], Y_validation)
        new_accuracy_shape = knn_shape.evaluate(X_validation[shape_mapping_validation], Y_validation)
        mean_accuracies += 0.5*(new_accuracy_rgb + new_accuracy_shape)
        print 'New accuracy_rgb', new_accuracy_rgb
        print 'New accuracy_shape', new_accuracy_shape

    mean_accuracies/=10
    print 'New mean Accuracy', mean_accuracies
    accuracies.append([K, mean_accuracies])

out = open('result_pickles/knn_accuracies_for_k.pickle', 'wb')
pickle.dump(accuracies, out)
out.close()

plt.xlabel('valor de K')
plt.ylabel('acuracia media')
x = [a for [a,b] in accuracies]
y = [b for [a,b] in accuracies]
plt.plot(x,y)
plt.show()
