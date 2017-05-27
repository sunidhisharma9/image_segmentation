import operator
import random
import numpy as np
import pandas as pd
import pickle

from classifiers.DataVectorizer import *
from classifiers.KnnClassifier import *
from classifiers.ClassifierTester import *

#Begin

print 'Testing RGB view'
dv = DataVectorizer(filename='result_pickles/RGB_view.pickle')
accuracies_RGB = []
for i in xrange(30):
    print 'iteration', i, 'of 30-------------'
    new_accuracies = ClassifierTester.make_n_fold_test(dv, 10, type_classifier='knn')
    accuracies_RGB += new_accuracies
    
print 'Testing Shape view'
dv = DataVectorizer(filename='result_pickles/shape_view.pickle')
accuracies_SHAPE = []
for i in xrange(30):
    print 'iteration', i, 'of 30-------------'
    new_accuracies = ClassifierTester.make_n_fold_test(dv, 10, type_classifier='knn')
    accuracies_SHAPE += new_accuracies

out_rgb = open('result_pickles/accuracies_RGB_knn.pickle', 'wb')
out_shape = open('result_pickles/accuracies_SHAPE_knn.pickle', 'wb')

pickle.dump(accuracies_RGB, out_rgb)
pickle.dump(accuracies_SHAPE, out_shape)

out_rgb.close()
out_shape.close()
