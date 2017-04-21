#By Felipe, Paulo e Debora

import pandas as pd
import numpy as np
import math as m
import pickle

COMPUTE_DISS = False

def read_from_csv(filename):
    raw_data = pd.read_csv(filename, sep=",", header=2)
    raw_data = raw_data.values #Numpy array
    return raw_data

def cvt_np_array(matrix):
    if type(matrix) != np.ndarray:
        matrix = np.array(matrix)
    return matrix

def print_data_overview(raw_data):    
    raw_data = cvt_np_array(raw_data)
    
    num_elems = np.shape(raw_data)[0]
    num_vars = np.shape(raw_data)[1]

    print 'Numero de elementos:', num_elems
    print 'Numero de variaveis:', num_vars

def euclid_dist(vect1, vect2):
    if len(vect1) != len(vect2):
        raise ValueError('The size of the vectors must be equal')

    vect1 = cvt_np_array(vect1)
    vect2 = cvt_np_array(vect2)
    
    diff = vect1 - vect2
    
    return np.sqrt(np.dot(diff, diff))

def calculate_diss_matrix(raw_data):
    raw_data = cvt_np_array(raw_data)
    
    num_elems = np.shape(raw_data)[0]
    diss_matrix = np.zeros((num_elems, num_elems))

    for i in xrange(num_elems):
        for j in xrange(num_elems):
            elem1 = raw_data[i]
            elem2 = raw_data[j]
        
            diss_matrix[i][j] = euclid_dist(elem1, elem2)    

    return diss_matrix

def create_views(raw_data):
    size_view_1 = 9
    size_view_2 = 10
    num_rows = np.shape(raw_data)[0]
    view_1 = raw_data[np.ix_(range(num_rows),range(size_view_1))]
    view_2 = raw_data[np.ix_(range(num_rows),range(size_view_1, size_view_2))]
    return [view_1, view_2]

#n is the number of elements to be clustered
#k is the number of clusters
#p is the number of views
#q is the number of elements of a cluster prototype
def initialize_clustering(n, k, p, q):
    U = np.ones((n,k))
    Lambda = np.ones((p, k))

    #shuffle indexes to randomly initialize cluster prototypes
    all_indexes = range(n)
    np.random.shuffle(all_indexes)

    G = np.zeros((k,q))

    for i in xrange(k):
        G[i] = np.array(all_indexes[i*q:(i+1)*q])
    
    return [U, Lambda, G]

#Begin

if COMPUTE_DISS:

    raw_data = read_from_csv('data/segmentation.test.txt')
    print_data_overview(raw_data)

    #Create the separated views
    [view_1, view_2] = create_views(raw_data)

    #Create the dissimilarity matrix for both views
    diss_matrix_1 = calculate_diss_matrix(view_1)
    diss_matrix_2 = calculate_diss_matrix(view_2)
else:
    print 'Loading previous matrices'
    diss_matrix_1 = pickle.load(open('diss_matrix_1.pickle','rb'))
    diss_matrix_2 = pickle.load(open('diss_matrix_1.pickle','rb'))
