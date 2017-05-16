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

##print 'Testing RGB view'
##dv = DataVectorizer(filename='RGB_view.pickle')
##tam_x = np.shape(dv.X)[0]
##all_data = []
##
##print 'Creating Data set'
##for i in xrange(tam_x):
##    all_data.append([dv.X[i], dv.Y[i]])
##
##random.shuffle(all_data)
##
##len_all_data = len(all_data)
##size_train = int(0.7*float(len_all_data))
##size_validation = int(0.15*float(len_all_data))
##size_test = int(0.15*float(len_all_data))
##
##train = all_data[:size_train]
##validation = all_data[size_train+1:size_train + size_validation]
##X_train = np.array([x for (x,y) in all_data])
##Y_train = np.array([y for (x,y) in all_data])
##X_validation = np.array([x for (x,y) in all_data])
##Y_validation = np.array([y for (x,y) in all_data])
##
##accuracies = []
##for i in xrange(49):
##    print 'Iteration', i, 'of 50'
##    K = 2*i + 1
##    knn = KnnClassifier(X_train, Y_train, K)
##    new_accuracy = knn.evaluate(X_validation, Y_validation)
##    print 'New Accuracy', new_accuracy
##    accuracies.append([K, new_accuracy])
##
##out = open('knn_accuracies.pickle', 'wb')
##pickle.dump(accuracies, out)
##out.close()


#Begin

def make_30_fold_test(data_vectorizer):
    X = data_vectorizer.X
    Y = data_vectorizer.Y
    classes = data_vectorizer.classes
    num_classes = len(classes)
    X_for_class = []
    X_index = [(index, x) for (index, x) in enumerate(X)]
    for i in xrange(num_classes):
        separated_X = [[x,i] for (index, x) in X_index if Y[index]==i]
        random.shuffle(separated_X)
        X_for_class.append(separated_X)

    folds = []
    for k in xrange(30):
        low_index = k*10
        high_index = low_index + 10
        new_fold = []
        for i in xrange(num_classes):
            new_fold += X_for_class[i][low_index:high_index]
        random.shuffle(new_fold)
        folds.append(new_fold)

    print 'Folds created'
    accuracies = []
    for k in xrange(30):
        test = folds[k]
        train = []
        for i in xrange(30):
            if i != k:
                train += folds[i]
        X_train = np.array([t[0] for t in train])
        Y_train = np.array([t[1] for t in train]).ravel()
        X_test = np.array([t[0] for t in test])
        Y_test = np.array([t[1] for t in test]).ravel()

        knn = KnnClassifier(X_train, Y_train, 1)
        new_accuracy = knn.evaluate(X_test, Y_test)
        print 'Accuracy', k, 'of 30', new_accuracy
        accuracies.append(new_accuracy)
    return accuracies

print 'Testing RGB view'
dv = DataVectorizer(filename='RGB_view.pickle')
accuracies_RGB = make_30_fold_test(dv)
print 'Testing Shape view'
dv = DataVectorizer(filename='shape_view.pickle')
accuracies_SHAPE = make_30_fold_test(dv)

out_rgb = open('accuracies_RGB_knn.pickle', 'wb')
out_shape = open('accuracies_SHAPE_knn.pickle', 'wb')

pickle.dump(accuracies_RGB, out_rgb)
pickle.dump(accuracies_SHAPE, out_shape)

out_rgb.close()
out_shape.close()
