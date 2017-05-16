from math import sqrt
import operator
import random
import numpy as np
import pandas as pd
import numpy as np
import pickle

class DataVectorizer:

    def __init__(self, pandas_data=None, filename=None):
        self.data_frame = None

        if not filename is None:
            self.get_from_pickle_pandas(filename)
        elif not pandas_data is None:
            self.data_frame = pandas_data

        if self.data_frame is None:
            raise ValueError('No input data defined')

        self.get_classes_from_data_frame()
        self.create_vectors()

    def get_from_pickle_pandas(self, filename):
        self.data_frame = pd.read_pickle(filename)

    def read_from_csv(self, filename):
        self.data_frame = pd.read_csv(filename, sep=",", header=2)

    def get_classes_from_data_frame(self):
        my_data = self.data_frame
        classes=[]
        for index, row in my_data.iterrows():
            classes.append(index)
        self.classes = sorted(set(classes))

    def create_vectors(self):
        X = []
        Y = []
        for (i, row) in self.data_frame.iterrows():
            X.append(row.values)
            current_class = self.classes.index(str(i))
            Y.append(current_class)
        self.X = np.array(X)
        self.Y = np.array(Y).ravel()

class KnnClassifier:
    def __init__(self, X, Y, K):
        self.X = X
        self.Y = Y
        self.K = K
        
    @staticmethod
    def euclid_dist(vect1, vect2):
        if len(vect1) != len(vect2):
            raise ValueError('The size of the vectors must be equal')

        vect1 = KnnClassifier.cvt_np_array(vect1)
        vect2 = KnnClassifier.cvt_np_array(vect2)
        
        diff = vect1 - vect2
        
        return np.sqrt(np.dot(diff, diff))

    @staticmethod
    def cvt_np_array(matrix):
        if type(matrix) != np.ndarray:
            matrix = np.array(matrix)
        return matrix
    
    def predict(self, x):
        distances = []
        X = self.X
        Y = self.Y
        size_X = np.shape(X)[0]
        
        for i in xrange(size_X):
            new_dist = KnnClassifier.euclid_dist(X[i], x)
            distances.append((X[i], Y[i], new_dist))

        distances_sorted = sorted(distances, key = lambda x: x[2])
        k_neighboors = [(e[0], e[1]) for e in distances_sorted[:self.K]]
        classes = {}
        for n in k_neighboors:
            if n[1] in classes:
                classes[n[1]] += 1
            else:
                classes[n[1]] = 1
        votes = sorted(classes.items(), key=operator.itemgetter(1), reverse=True)
        return votes[0]

    def evaluate(self, X, Y):
        num_total = 0.0
        num_right = 0.0
        num_wrong = 0.0

        for (index, row) in enumerate(X):
            current_index = Y[index]
            row_values = row
            predicted_index = self.predict(row_values)[0]
            if predicted_index == current_index:
                num_right += 1.0
            else:
                num_wrong += 1.0
            num_total += 1.0

        return num_right/num_total

print 'Testing RGB view'
dv = DataVectorizer(filename='RGB_view.pickle')
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
X_train = np.array([x for (x,y) in all_data])
Y_train = np.array([y for (x,y) in all_data])
X_validation = np.array([x for (x,y) in all_data])
Y_validation = np.array([y for (x,y) in all_data])

accuracies = []
for i in xrange(49):
    print 'Iteration', i, 'of 50'
    K = 2*i + 1
    knn = KnnClassifier(X_train, Y_train, K)
    new_accuracy = knn.evaluate(X_validation, Y_validation)
    print 'New Accuracy', new_accuracy
    accuracies.append([K, new_accuracy])

out = open('knn_accuracies.pickle', 'wb')
pickle.dump(accuracies, out)
out.close()
