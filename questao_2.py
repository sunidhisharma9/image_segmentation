# By Felipe {fnba}, Paulo {pan2} e Debora {dca2}@cin.ufpe.br

import pandas as pd
import numpy as np
import csv
import pickle
import warnings

# ignore warnings
warnings.filterwarnings('ignore')


class BayesClassifier:
    def __init__(self, read_files=False):
        self.read_files = read_files
        self.classes = ['BRICKFACE', 'SKY', 'FOLIAGE', 'CEMENT', 'WINDOW', 'PATH', 'GRASS']

        if read_files:
            print ("Reading input file")
            self.read_from_csv()
            self.print_data_overview(self.raw_data)

    def read_from_csv(self):
        rd = pd.read_csv("data/segmentation.test.txt", sep=",", header=2)
        self.data_frame = rd
        rd = rd.values  # Numpy array
        self.raw_data = rd

    def run_test(self):
        self.read_from_csv()
        self.get_classes()

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
        with open("data/segmentation.test.txt") as csvfile:
            reader = csv.DictReader(csvfile,)

    def get_classes(self):
        mensagem = 'Coletando classe: '
        w=self.classes
        w_brickface = []
        w_sky = []
        w_foliage = []
        w_cement = []
        w_window = []
        w_path = []
        w_grass = []
        df = self.data_frame

        for index, row in df.iterrows():
            if str(index).__eq__(w[0]):
                w_brickface.append(row)
                print mensagem, w[0]
            if str(index).__eq__(w[1]):
                w_sky.append(row)
                print mensagem, w[1]
            if str(index).__eq__ (w[2]):
                w_foliage.append(row)
                print mensagem, w[2]
            if str(index).__eq__(w[3]):
                w_cement.append(row)
                print mensagem, w[3]
            if str(index).__eq__(w[4]):
                w_window.append(row)
                print mensagem, w[4]
            if str(index).__eq__(w[5]):
                w_path.append(row)
                print mensagem, w[5]
            if str(index).__eq__(w[6]):
                w_grass.append(row)
                print mensagem, w[6]

        self.persist(w_brickface, w[0].lower() + '.pickle')
        self.persist(w_sky, w[1].lower() + '.pickle')
        self.persist(w_cement, w[2].lower() + '.pickle')
        self.persist(w_brickface, w[3].lower() + '.pickle')
        self.persist(w_window, w[4].lower() + '.pickle')
        self.persist(w_path, w[5].lower() + '.pickle')
        self.persist(w_grass, w[6].lower() + '.pickle')
        self.persist(df,str(w.__class__)+'.pickle')

    def persist(self, to_persist, file_name):
        file = open(file_name,'wb')
        pickle.dump(to_persist,file)
        file.close()

    def gets_w_classes(self,file_w_class_name):
        file = open(file_w_class_name,'r')
        if file != None:
            return True
        else:
            return False

    # Begin
bc = BayesClassifier(True)
bc.run_test()
