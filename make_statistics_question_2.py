import matplotlib.pyplot as plt
import pickle
import numpy as np

rgb_accuracies = pickle.load(open('result_pickles/accuracies_RGB.pickle', 'rb'))
shape_accuracies = pickle.load(open('result_pickles/accuracies_SHAPE.pickle', 'rb'))

rgb_np = np.array(rgb_accuracies)
shape_np = np.array(shape_accuracies)

def create_IC_interval(np_array):
    t_29_0475 = 2.0452
    len_array = len(np_array)
    mean_array = np.mean(np_array)
    std_array = np.std(np_array, ddof=1)
    int_lower_array = mean_array - t_29_0475*std_array/np.sqrt(len_array)
    int_high_array = mean_array + t_29_0475*std_array/np.sqrt(len_array)
    return [mean_array, int_lower_array, int_high_array]

print 'BAYES-----------'

ic_bayes_rgb = create_IC_interval(rgb_np)
print 'RGB'
print 'Mean', ic_bayes_rgb[0]
print 'IC 95 [', ic_bayes_rgb[1], ',' , ic_bayes_rgb[2], ']'

ic_bayes_shape = create_IC_interval(shape_np)
print 'SHAPE'
print 'Mean', ic_bayes_shape[0]
print 'IC 95 [', ic_bayes_shape[1], ',' , ic_bayes_shape[2], ']'
