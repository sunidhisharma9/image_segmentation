# By Felipe {fnba}, Paulo {pan2} e Debora {dca2}@cin.ufpe.br
from pandas import index

import pandas as pd
import numpy as np
import csv
import pickle
import warnings

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
            self.data = self.get_from_pickle_pandas('rgb_view.pickle')
        else:
            self.data = self.get_from_pickle_pandas('shape_view.pickle')

        self.classes = self.get_classes_dinamicamente()

        print 'Getting a priori probabilities'
        self.get_w_frequenz()

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
        w = self.classes
        df = self.get_from_pickle_pandas('rgb_view.pickle') #self.data_frame
        w_frequenz = None
        num_elems = np.shape(df)[0]

        qtd_w_brickface = 0
        qtd_w_sky = 0
        qtd_w_foliage = 0
        qtd_w_cement = 0
        qtd_w_window = 0
        qtd_w_path = 0
        qtd_w_grass = 0

        for index, row in df.iterrows():
            if str(index).__eq__(w[0]):
                qtd_w_grass  += 1
            if str(index).__eq__(w[1]):
                qtd_w_path += 1
            if str(index).__eq__ (w[2]):
                qtd_w_window += 1
            if str(index).__eq__(w[3]):
                qtd_w_cement += 1
            if str(index).__eq__(w[4]):
                qtd_w_foliage += 1
            if str(index).__eq__(w[5]):
                qtd_w_sky += 1
            if str(index).__eq__(w[6]):
                qtd_w_brickface += 1

        w_frequenz = np.array([
            [w[0], qtd_w_grass, self.divide_probability(qtd_w_grass,num_elems)],
             [w[1], qtd_w_path, self.divide_probability(qtd_w_path,num_elems)],
             [w[2], qtd_w_window, self.divide_probability(qtd_w_window,num_elems)],
             [w[3], qtd_w_cement, self.divide_probability(qtd_w_cement,num_elems)],
             [w[4], qtd_w_foliage, self.divide_probability(qtd_w_foliage,num_elems)],
             [w[5], qtd_w_sky, self.divide_probability(qtd_w_sky,num_elems)],
             [w[6], qtd_w_brickface, self.divide_probability(qtd_w_brickface,num_elems)]])
        self.persist(w_frequenz,'w_frequence.pickle')
        print w_frequenz

    def divide_probability(self,value_a, value_b):
        result = round(float(value_a/float(value_b)), 2)
        return result

    def separarViews(self):
        size_view_1 = 9
        size_view_2 = 10
        columns = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        file = None
        #with open("data/segmentation.test.txt","rb") as csv_with_headers:
        file = pd.read_csv("data/segmentation.test.txt","rb",header=2) #csv.reader(csv_with_headers)
         #  for index, line in enumerate(reader):
        print file

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

    def get_max_verossimilhanca(self):
        df = self.data_frame
        w_grass_max = 0
        w_path_max = 0
        w_brickface_max = 0
        w_sky_max = 0
        w_window_max = 0
        w_foliage_max = 0
        w_cement_max = 0

        #get grass pickled file
        df_grass = self.get_from_pickle_pandas("grass.pickle")
        np_grass_array = self.pickled_dataframe_to_numpy_array(df_grass)
        w_grass_max = self.process_max(np_grass_array)

        # get path_pickled file
        df_path = self.get_from_pickle_pandas("path.pickle")
        np_path_array = self.pickled_dataframe_to_numpy_array(df_path)
        w_path_max = self.process_max(np_path_array)

        # get brickface pickled file)
        df_brickface = self.get_from_pickle_pandas("brickface.pickle")
        np_brickface_array = self.pickled_dataframe_to_numpy_array(df_brickface)
        w_brickface_max = self.process_max(np_brickface_array)

        #get sky pickled file
        df_sky = self.get_from_pickle_pandas("sky.pickle")
        np_sky_array = self.pickled_dataframe_to_numpy_array(df_sky)
        w_sky_max = self.process_max(np_sky_array)

        # get window pickled file
        df_window = self.get_from_pickle_pandas("window.pickle")
        np_window_array = self.pickled_dataframe_to_numpy_array(df_window)
        w_window_max = self.process_max(np_window_array)

        # get foliage pickled file
        df_foliage = self.get_from_pickle_pandas("foliage.pickle")
        np_foliage_array = self.pickled_dataframe_to_numpy_array(df_foliage)
        w_window_max = self.process_max(np_foliage_array)

        # get cement pickled file
        df_cement = self.get_from_pickle_pandas("cement.pickle")
        np_cement_array = self.pickled_dataframe_to_numpy_array(df_cement)
        w_cement_max = self.process_max(np_cement_array)



    def process_max(self, np_array):
        i = 0  # linha
        j = 0  # coluna
        max = 0
        old_value = 0
        num_of_rows = np.shape(np_array)[0]
        np_array = np_array.values

        while i < num_of_rows:
            old_value = max
            if max < np.amax(np_array[i]):
                max = np.amax(np_array[i])
                print 'New maximal value found', 'old value: ', old_value, 'new value', max
            i = i + 1
        return max


    def pickled_dataframe_to_numpy_array(self, data_frame):
        new_np_array = pd.DataFrame(data_frame)
        return new_np_array

    @staticmethod
    def get_from_pickle_pandas(file_name):
        df = pd.read_pickle(file_name)
        return df

    def bayes(self):
        df_frequenz = self.get_from_pickle_pandas('w_frequenz.pickle')
        class_name = None
        w_collection = None
        E = 0
        Pwj=0
        for w in self.classes:
            w_collection = self.get_from_pickle_pandas(class_name)
            w_np_array = self.pickled_dataframe_to_numpy_array(w_collection)
            E = np.sum(w_collection)
            i=0
           # for j in w_collection:
           #     E +=w_collection[j]
        #x = np.shape(self.raw_data)[1] #0,14
        #w = df_frequenz[0][1] #300

        # somatorio
        #p_x_w = None
        max_w = None

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
bc = BayesClassifier(True)