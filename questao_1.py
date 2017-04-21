#By Felipe, Paulo e Debora

import pandas as pd
import numpy as np
import math as m
import pickle

class ClusterMaker:

    
    #n is the number of elements to be clustered
    #k is the number of clusters
    #p is the number of views
    #q is the number of elements of a cluster prototype
    def __init__(self, filename, k, q, m, s, read_files=False):
        self.k = k
        self.q = q
        self.p = 2
        self.m = m
        self.s = s
        
        if read_files:
            print 'Reading input file'
            self.read_from_csv(filename)
            ClusterMaker.print_data_overview(self.raw_data)

            #Create the separated views
            self.create_views()

            #Create the dissimilarity matrix for both views
            self.diss_matrix_1 = ClusterMaker.calculate_diss_matrix(self.view_1)
            self.diss_matrix_2 = ClusterMaker.calculate_diss_matrix(self.view_2)
        else:
            print 'Loading previous matrices'
            self.diss_matrix_1 = pickle.load(open('diss_matrix_1.pickle','rb'))
            self.diss_matrix_2 = pickle.load(open('diss_matrix_1.pickle','rb'))

        self.diss = [self.diss_matrix_1, self.diss_matrix_2]
        
        print 'Dissimilarity matrices calculated'
        self.n = np.shape(self.diss_matrix_1)[0]
        
    def run_clustering(self):
        self.initialize_clustering()

    def read_from_csv(self, filename):
        rd = pd.read_csv(filename, sep=",", header=2)
        rd = rd.values #Numpy array
        self.raw_data = rd

    def create_views(self):
        size_view_1 = 9
        size_view_2 = 10
        num_rows = np.shape(self.raw_data)[0]
        self.view_1 = self.raw_data[np.ix_(range(num_rows),range(size_view_1))]
        self.view_2 = self.raw_data[np.ix_(range(num_rows),range(size_view_1, size_view_2))]

    def initialize_clustering(self):
        n = self.n
        k = self.k
        p = self.p
        q = self.q
        
        self.U = np.ones((n, k))
        self.Lambda = np.ones((p, k))

        #shuffle indexes to randomly initialize cluster prototypes
        all_indexes = range(n)
        np.random.shuffle(all_indexes)

        self.G = np.zeros((k, q))

        for i in xrange(self.k):
            self.G[i] = np.array(all_indexes[i*q:(i+1)*q])
        

    def dist_to_cluster(self, elem_index, cluster_index):
        sum = 0
        for p in xrange(self.p):
            partial_sum = 0
            for j in xrange(self.q):
                cluster = self.G[cluster_index]
                partial_sum += self.diss[p][elem_index, cluster[j]]
            sum += self.Lambda[p][cluster_index]*partial_sum
        return sum

    def cost(self):
        sum = 0
        for k in xrange(self.k):
            for i in xrange(self.n):
                sum += pow(self.U[i][k],self.m)*self.dist_to_cluster(i, k)
        return sum
    
    @staticmethod
    def cvt_np_array(matrix):
        if type(matrix) != np.ndarray:
            matrix = np.array(matrix)
        return matrix

    @staticmethod
    def print_data_overview(raw_data):    
        raw_data = ClusterMaker.cvt_np_array(raw_data)
        
        num_elems = np.shape(raw_data)[0]
        num_vars = np.shape(raw_data)[1]

        print 'Numero de elementos:', num_elems
        print 'Numero de variaveis:', num_vars

    @staticmethod
    def euclid_dist(vect1, vect2):
        if len(vect1) != len(vect2):
            raise ValueError('The size of the vectors must be equal')

        vect1 = ClusterMaker.cvt_np_array(vect1)
        vect2 = ClusterMaker.cvt_np_array(vect2)
        
        diff = vect1 - vect2
        
        return np.sqrt(np.dot(diff, diff))

    @staticmethod
    def calculate_diss_matrix(raw_data):
        raw_data = ClusterMaker.cvt_np_array(raw_data)
        
        num_elems = np.shape(raw_data)[0]
        diss_matrix = np.zeros((num_elems, num_elems))

        for i in xrange(num_elems):
            for j in xrange(num_elems):
                elem1 = raw_data[i]
                elem2 = raw_data[j]
            
                diss_matrix[i][j] = ClusterMaker.euclid_dist(elem1, elem2)    

        return diss_matrix
    
#Begin

cm = ClusterMaker('data/segmentation.data.txt', 7, 3, 1.6, 1, read_files=True) 
cm.run_clustering()
