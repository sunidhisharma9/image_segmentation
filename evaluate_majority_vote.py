# By Felipe {fnba}, Paulo {pan2} e Debora {dca2}@cin.ufpe.br
from pandas import index

import pandas as pd
import numpy as np
import csv
import pickle
import warnings
import random

from classifiers.DataVectorizer import *
from classifiers.MajorityVoteClassifier import *
from classifiers.ClassifierTester import *

print 'Testing full view'
dv = DataVectorizer(filename='result_pickles/full_view.pickle')
accuracies_full = []
for i in xrange(30):
    print 'iteration', i, 'of 30-------------'
    new_accuracies = ClassifierTester.make_n_fold_test(dv, 10, type_classifier='majority')
    accuracies_full += new_accuracies

out_full = open('result_pickles/accuracies_majority_full.pickle', 'wb')

pickle.dump(accuracies_full, out_full)

out_full.close()