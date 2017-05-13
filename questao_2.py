# By Felipe {fnba}, Paulo {pan2} e Debora {dca2}@cin.ufpe.br
from pandas import index

import pandas as pd
import numpy as np
import csv
import pickle
import warnings

#benchmark
from sklearn.naive_bayes import GaussianNB

# ignore warnings
warnings.filterwarnings('ignore')


class BayesClassifier:
    def __init__(self, read_files=False, view_to_use=1):
        self.read_files = read_files

        if read_files:
            print ("Loading view to use")
            self.read_from_csv()
            self.create_views()
            self.print_data_overview(self.raw_data)

        if view_to_use == 1:
            print 'Using the rgb view'
            self.data = self.get_from_pickle_pandas('rgb_view.pickle')
        else:
            print 'Using the shape view'
            self.data = self.get_from_pickle_pandas('shape_view.pickle')

        self.classes = self.get_classes_dinamicamente()

        #Build benchmark
        X = []
        Y = []
        for (i, row) in self.data.iterrows():
            X.append(row.values)
            current_class = self.classes.index(str(i))
            Y.append(current_class)
        X_np = np.array(X)
        Y_np = np.array(Y).ravel()

        benchmark_classifier = GaussianNB()
        benchmark_classifier.fit(X_np, Y_np)
        print 'Banchmark accuracy', benchmark_classifier.score(X_np, Y_np)

        print 'Getting a priori probabilities'
        self.get_w_frequenz()
        print 'Calculate the probabilities distributions by the max likelihood method'
        self.calculate_prob_diss_classes()
        print 'Our classifier'
        print 'Accuracy', self.evaluate()
    def read_from_csv(self):
        rd = pd.read_csv("data/segmentation.test.txt", sep=",", header=2)
        self.data_frame = rd
        rd = rd.values  # Numpy array
        self.raw_data = rd

    def run_test(self):
        self.read_from_csv()
        self.get_w_frequenz()

    @staticmethod
    def print_data_overview(raw_data):
        raw_data = BayesClassifier.cvt_np_array(raw_data)
        num_elems = np.shape(raw_data)[0]
        num_vars = np.shape(raw_data)[1]

        print 'Numero de elementos:', num_elems
        print 'Numero de variaveis:', num_vars

    @staticmethod
    def cvt_np_array(matrix):
        if type(matrix) != np.ndarray:
            matrix = np.array(matrix)
        return matrix

    def read_from_csv_with_headers(self):
        with open("data/segmentation.test.txt") as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                print row


    def persist(self, to_persist, file_name):
        file = open(file_name,'wb')
        pickle.dump(to_persist,file)
        file.close()

    #checks if the named file exists
    def gets_w_classes(self,file_w_class_name):
        file = open(file_w_class_name,'wb')
        if file is not None:
            return True
        else:
            return False

    def get_w_frequenz(self):
        my_classes = self.classes
        data_frame = self.data
        num_classes = len(my_classes)
        num_elems = np.shape(data_frame)[0]

        self.apriori = np.zeros((num_classes, 1))

        for index, row in data_frame.iterrows():
            current_class = str(index)
            current_index = my_classes.index(current_class)
            self.apriori[current_index] += 1

        self.apriori /= num_elems

    def calculate_prob_diss_classes(self):
        my_classes = self.classes
        data_frame = self.data
        num_classes = len(my_classes)
        num_elems = np.shape(data_frame)[0]
        element_size = np.shape(data_frame)[1]
        self.centers = np.zeros((num_classes, element_size))
        self.num_for_class = np.zeros((num_classes, 1))
        self.diags = np.zeros((num_classes, element_size))

        #Compute centers
        for index, row in data_frame.iterrows():
            current_class = str(index)
            current_index = my_classes.index(current_class)
            self.centers[current_index] += row.values
            self.num_for_class[current_index] += 1
        self.centers /= self.num_for_class

        #Compute variance matrix diagonals
        for index, row in data_frame.iterrows():
            current_class = str(index)
            current_index = my_classes.index(current_class)
            current_center = self.centers[current_index]
            self.diags[current_index] += pow(row.values - current_center, 2)

        self.diags /= self.num_for_class


    def p_x_w(self, x, index):
        d = len(x)
        center = self.centers[index]
        variance_diag = self.diags[index]
        prod = np.prod(variance_diag)
        diff_2 = pow(x - center,2)
        diff_2_div = diff_2/variance_diag
        internal_sum = np.sum(diff_2_div)
        return pow(2*np.pi, -0.5*d)*pow(prod, -0.5)*np.exp((-0.5)*internal_sum)

    def p_w_x(self, x, index):
        probs = np.array([self.p_x_w(x, i) for (i, c) in enumerate(self.classes)])
        return probs[index]*self.apriori[index]/np.sum(probs)

    def predict(self, x):
        probs = np.array([self.p_w_x(x, index) for (index, c) in enumerate(self.classes)])
        max_index = np.argmax(probs)
        return [max_index, self.classes[max_index]]

    def evaluate(self):
        my_classes = self.classes
        data_frame = self.data
        num_total = 0.0
        num_right = 0.0
        num_wrong = 0.0

        for (index, row) in data_frame.iterrows():
            current_class = str(index)
            current_index = my_classes.index(current_class)
            row_values = row.values
            predicted_index = self.predict(row_values)[0]
            if predicted_index == current_index:
                num_right += 1.0
            else:
                num_wrong += 1.0
            num_total += 1.0

        print num_right/num_total

    def create_views(self):
        shape_columns=["REGION-CENTROID-COL", "REGION-CENTROID-ROW", "REGION-PIXEL-COUNT", "SHORT-LINE-DENSITY-5", "SHORT-LINE-DENSITY-2", "VEDGE-MEAN", "VEDGE-SD", "HEDGE-MEAN", "HEDGE-SD", "INTENSITY-MEAN"]
        rgb_columns=["RAWRED-MEAN", "RAWBLUE-MEAN", "RAWGREEN-MEAN", "EXRED-MEAN", "EXBLUE-MEAN", "EXGREEN-MEAN", "VALUE-MEAN", "SATURATION-MEAN","HUE-MEAN"]
        df = self.data_frame
        shape_view = df[shape_columns].copy()
        self.persist(shape_view, "shape_view.pickle")
        self.shape_view = shape_view
        rgb_view = df[rgb_columns].copy()
        self.persist(rgb_view,"rgb_view.pickle")
        self.rgb_view = rgb_view

    def pickled_dataframe_to_numpy_array(self, data_frame):
        new_np_array = pd.DataFrame(data_frame)
        return new_np_array

    @staticmethod
    def get_from_pickle_pandas(file_name):
        df = pd.read_pickle(file_name)
        return df

    def get_classes_dinamicamente(self):
        colecao = self.data
        classes=[]
        class_name=""
        index_number = 1
        j = 0
        for index, row in colecao.iterrows():
            classes.append(index)
        classes = sorted(set(classes))
        print 'Detected classes', classes
        return classes





#Begin
#bc = BayesClassifier(True)