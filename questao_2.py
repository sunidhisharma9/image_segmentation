# By Felipe {fnba}, Paulo {pan2} e Debora {dca2}@cin.ufpe.br
from pandas import index

import pandas as pd
import numpy as np
import csv
import pickle
import warnings
import random

from KnnClassifier import *

# ignore warnings
warnings.filterwarnings('ignore')


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
        classes = []
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


# ---------------


class BayesClassifier:
    def __init__(self, X, Y, classes):
        self.classes = classes
        self.X = X
        self.Y = Y

        print 'Getting a priori probabilities'
        self.get_w_frequency()
        print 'Calculate the probabilities distributions by the max likelihood method'
        self.calculate_prob_diss_classes()

    @staticmethod
    def cvt_np_array(matrix):
        if type(matrix) != np.ndarray:
            matrix = np.array(matrix)
        return matrix

    @staticmethod
    def read_from_csv_with_headers():
        with open("data/segmentation.test.txt") as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                print row

    def persist(self, to_persist, file_name):
        file = open(file_name, 'wb')
        pickle.dump(to_persist, file)
        file.close()

    # checks if the named file exists
    def gets_w_classes(self, file_w_class_name):
        file = open(file_w_class_name, 'wb')
        if file is not None:
            return True
        else:
            return False

    def get_w_frequency(self):
        my_classes = self.classes
        num_classes = len(my_classes)
        num_elems = np.shape(self.X)[0]

        self.apriori = np.zeros((num_classes, 1))

        for index, row in enumerate(self.X):
            current_index = self.Y[index]
            self.apriori[current_index] += 1

        self.apriori /= num_elems

    def calculate_prob_diss_classes(self):
        my_classes = self.classes
        num_classes = len(my_classes)

        element_size = np.shape(self.X)[1]

        self.centers = np.zeros((num_classes, element_size))
        self.num_for_class = np.zeros((num_classes, 1))
        self.diags = np.zeros((num_classes, element_size))

        # Compute centers
        for index, row in enumerate(self.X):
            current_index = self.Y[index]
            self.centers[current_index] += row
            self.num_for_class[current_index] += 1
        self.centers /= self.num_for_class

        # Compute variance matrix diagonals
        for index, row in enumerate(self.X):
            current_index = self.Y[index]
            current_center = self.centers[current_index]
            self.diags[current_index] += pow(row - current_center, 2)

        self.diags /= self.num_for_class

        for c in xrange(num_classes):
            for d in xrange(element_size):
                if self.diags[c][d] == 0:
                    self.diags[c][d] = 0.0000000001  # Avoid zero divisions

    def p_x_w(self, x, index):
        d = len(x)
        center = self.centers[index]
        variance_diag = self.diags[index]
        prod = np.prod(variance_diag)
        diff_2 = pow(x - center, 2)
        diff_2_div = diff_2 / variance_diag
        internal_sum = np.sum(diff_2_div)
        return pow(2 * np.pi, -0.5 * d) * pow(prod, -0.5) * np.exp((-0.5) * internal_sum)

    def p_w_x(self, x, index):
        probs = np.array([self.p_x_w(x, i) for (i, c) in enumerate(self.classes)])
        return probs[index] * self.apriori[index] / np.sum(probs)

    def predict(self, x):
        probs = np.array([self.p_w_x(x, index) for (index, c) in enumerate(self.classes)])
        max_index = np.argmax(probs)
        return [max_index, self.classes[max_index]]

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

        return num_right / num_total

    def create_views(self):
        shape_columns = ["REGION-CENTROID-COL", "REGION-CENTROID-ROW", "REGION-PIXEL-COUNT", "SHORT-LINE-DENSITY-5",
                         "SHORT-LINE-DENSITY-2", "VEDGE-MEAN", "VEDGE-SD", "HEDGE-MEAN", "HEDGE-SD", "INTENSITY-MEAN"]
        rgb_columns = ["RAWRED-MEAN", "RAWBLUE-MEAN", "RAWGREEN-MEAN", "EXRED-MEAN", "EXBLUE-MEAN", "EXGREEN-MEAN",
                       "VALUE-MEAN", "SATURATION-MEAN", "HUE-MEAN"]
        df = self.data_frame
        shape_view = df[shape_columns].copy()
        self.persist(shape_view, "shape_view.pickle")
        self.shape_view = shape_view
        rgb_view = df[rgb_columns].copy()
        self.persist(rgb_view, "rgb_view.pickle")
        self.rgb_view = rgb_view


# Begin

def make_30_fold_test(data_vectorizer):
    X = data_vectorizer.X
    Y = data_vectorizer.Y
    classes = data_vectorizer.classes
    num_classes = len(classes)
    X_for_class = []
    X_index = [(index, x) for (index, x) in enumerate(X)]
    for i in xrange(num_classes):
        separated_X = [[x, i] for (index, x) in X_index if Y[index] == i]
        random.shuffle(separated_X)
        X_for_class.append(separated_X)

    folds = []
    for k in xrange(30):
        low_index = k * 10
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

        bc = BayesClassifier(X_train, Y_train, classes)
        new_accuracy = bc.evaluate(X_test, Y_test)
        print 'Accuracy', k, 'of 30', new_accuracy
        accuracies.append(new_accuracy)
    return accuracies


def friedman_test(dv):

    y = dv.Y
    #Sorting from minimal to maximal values
    x = np.sort(dv.X)
    classes = dv.classes
 #   bc = BayesClassifier(x, y, classes)
 #   knn = KnnClassifier(x, y, 7)
 #   knn_votes = knn.evaluate(x,y)
 #   bayes_result = bc.evaluate(x, y)
    knn_result = 0
    num_of_rows = len(y)
    num_of_columns = len(x[0])
    num_of_classes = len(classes)
    new_ndarray = np.zeros(shape=(num_of_rows, num_of_columns+1))
    for w_tmp in xrange(num_of_rows):
        new_ndarray[w_tmp][0] = y[w_tmp]
        for x_tmp in xrange(num_of_columns):
            new_ndarray[w_tmp][x_tmp+1] = x[w_tmp][x_tmp]
    # Creating an empty rank
    rank = create_empty_rank(num_of_classes=num_of_classes)
    rank_length=rank.__len__()
    sum_x = 0
    for new_w in xrange(num_of_classes):
        for w in xrange(num_of_rows):
            if y[w] == new_ndarray[w][0]:
                for x in xrange(num_of_columns):
                    #forwarding from w (class) column
                    x += 1
                    if w < num_of_rows - 1:
                        if y[w + 1] != y[w]:
                            sum_x += new_ndarray[w][x]
                            sum_x = put_to_rank(classes=classes, rank=rank,rank_length=rank_length,sum_x=sum_x,w=w,y=y)
                        else:
                            sum_x += new_ndarray[w][x]
                    else:
                        sum_x = put_to_rank(classes, rank,rank_length, sum_x, w, y)  #new_ndarray[w][x]

    print rank
    print 'Done'

#puts values to rank and returns a new counter
def put_to_rank(classes, rank, rank_length, sum_x, w, y):
    for g in xrange(rank_length - 1):
        for k in xrange(rank[g].__len__()):
            if rank[g][k - 1] == y[w]:
                print 'Ranking class ', classes[y[w] - 1], sum_x
                rank[1][k - 1] += sum_x
                print 'Preparing for next class...'
                sum_x = 0
    return sum_x

# Returns an empty rank
def create_empty_rank(num_of_classes):
    rank = np.zeros(shape=(2, num_of_classes))
    # defining the groups
    for n in xrange(num_of_classes):
        rank[0][n] = n
    return rank


#print 'Testing RGB view'
dv = DataVectorizer(filename='RGB_view.pickle')
accuracies_RGB = make_30_fold_test(dv)
friedman_test(dv)
#print 'Testing Shape view'
#dv = DataVectorizer(filename='shape_view.pickle')
#accuracies_SHAPE = make_30_fold_test(dv)



out_rgb = open('accuracies_RGB.pickle', 'wb')
out_shape = open('accuracies_SHAPE.pickle', 'wb')

pickle.dump(accuracies_RGB, out_rgb)
#pickle.dump(accuracies_SHAPE, out_shape)

out_rgb.close()
out_shape.close()
