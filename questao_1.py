#By Felipe, Paulo e Debora

import pandas as pd
import numpy as np
import math as m

def read_from_csv(filename):
    raw_data = pd.read_csv(filename, sep=",", header=2)
    raw_data = raw_data.values #Esse eh um array do numpy
    return raw_data

def print_data_overview(raw_data):    
    if type(raw_data) != np.ndarray:
        raw_data = np.array(raw_data)
    
    num_elems = np.shape(raw_data)[0]
    num_vars = np.shape(raw_data)[1]

    print 'Numero de elementos:', num_elems
    print 'Numero de variaveis:', num_vars

def euclid_dist(vect1, vect2):
    if len(vect1) != len(vect2):
        raise ValueError('The size of the vectors must be equal')

    if type(vect1) != np.ndarray:
        vect1 = np.array(vect1)

    if type(vect2) != np.ndarray:
        vect2 = np.array(vect2)
    
    diff = vect1 - vect2
    
    return np.sqrt(np.dot(diff, diff))

def calculate_diss_matrix(raw_data):
    if type(raw_data) != np.ndarray:
        raw_data = np.array(raw_data)
    
    num_elems = np.shape(raw_data)[0]
    diss_matrix = np.zeros((num_elems, num_elems))

    for i in xrange(num_elems):
        for j in xrange(num_elems):
            elem1 = raw_data[i]
            elem2 = raw_data[j]
        
            diss_matrix[i][j] = euclid_dist(elem1, elem2)    

    return diss_matrix

#Begin

raw_data = read_from_csv('data/segmentation.data.txt')
print_data_overview(raw_data)
diss_matrix = calculate_diss_matrix(raw_data)
