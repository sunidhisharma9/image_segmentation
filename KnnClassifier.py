import operator
import numpy as np
import pandas as pd
import pickle

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
        return votes[0][0]

    def evaluate(self, X, Y):
        num_total = 0.0
        num_right = 0.0
        num_wrong = 0.0

        for (index, row) in enumerate(X):
            current_index = Y[index]
            row_values = row
            predicted_index = self.predict(row_values)
            if predicted_index == current_index:
                num_right += 1.0
            else:
                num_wrong += 1.0
            num_total += 1.0

        return num_right/num_total