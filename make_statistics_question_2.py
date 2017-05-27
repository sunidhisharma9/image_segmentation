import matplotlib.pyplot as plt
import pickle
import numpy as np

rgb_accuracies_bayes = pickle.load(open('result_pickles/accuracies_RGB.pickle', 'rb'))
shape_accuracies_bayes = pickle.load(open('result_pickles/accuracies_SHAPE.pickle', 'rb'))

rgb_accuracies_knn = pickle.load(open('result_pickles/accuracies_RGB_knn.pickle', 'rb'))
shape_accuracies_knn = pickle.load(open('result_pickles/accuracies_SHAPE_knn.pickle', 'rb'))

accuracies_majority = pickle.load(open('result_pickles/accuracies_majority_full.pickle', 'rb'))

rgb_np_bayes = np.array(rgb_accuracies_bayes)
shape_np_bayes = np.array(shape_accuracies_bayes)

rgb_np_knn = np.array(rgb_accuracies_knn)
shape_np_knn = np.array(shape_accuracies_knn)

majority_np = np.array(accuracies_majority)

def create_IC_interval(np_array):
    t_29_0475 = 2.0452
    len_array = len(np_array)
    mean_array = np.mean(np_array)
    std_array = np.std(np_array, ddof=1)
    int_lower_array = mean_array - t_29_0475*std_array/np.sqrt(len_array)
    int_high_array = mean_array + t_29_0475*std_array/np.sqrt(len_array)
    return [mean_array, int_lower_array, int_high_array]

print 'BAYES-----------'

ic_bayes_rgb = create_IC_interval(rgb_np_bayes)
print 'RGB'
print 'Mean', ic_bayes_rgb[0]
print 'IC 95 [', ic_bayes_rgb[1], ',' , ic_bayes_rgb[2], ']'

ic_bayes_shape = create_IC_interval(shape_np_bayes)
print 'SHAPE'
print 'Mean', ic_bayes_shape[0]
print 'IC 95 [', ic_bayes_shape[1], ',' , ic_bayes_shape[2], ']'

print 'KNN-----------'

ic_knn_rgb = create_IC_interval(rgb_np_knn)
print 'RGB'
print 'Mean', ic_knn_rgb[0]
print 'IC 95 [', ic_knn_rgb[1], ',' , ic_knn_rgb[2], ']'

ic_knn_shape = create_IC_interval(shape_np_knn)
print 'SHAPE'
print 'Mean', ic_knn_shape[0]
print 'IC 95 [', ic_knn_shape[1], ',' , ic_knn_shape[2], ']'

print 'MAJORITY-----------'

ic_majority = create_IC_interval(majority_np)
print 'Mean', ic_majority[0]
print 'IC 95 [', ic_majority[1], ',' , ic_majority[2], ']'
