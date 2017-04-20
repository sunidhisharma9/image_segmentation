#By Felipe, Paulo e Debora

import pandas as pd
import numpy as np
import math as m

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

#Begin

raw_data = read_from_csv('data/segmentation.data.txt')
print_data_overview(raw_data)

#Create the separated views
[view_1, view_2] = create_views(raw_data)

#Create the dissimilarity matrix for both views
diss_matrix_1 = calculate_diss_matrix(view_1)
diss_matrix_2 = calculate_diss_matrix(view_2)
