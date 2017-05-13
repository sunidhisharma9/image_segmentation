import matplotlib.pyplot as plt
import pickle
import numpy as np

rgb_accuracies = pickle.load(open('accuracies_RGB.pickle', 'rb'))
shape_accuracies = pickle.load(open('accuracies_SHAPE.pickle', 'rb'))

rgb_np = np.array(rgb_accuracies)
shape_np = np.array(shape_accuracies)

t_29_0475 = 2.0452

mean_rgb = np.mean(rgb_np)
std_rgb = np.std(rgb_np, ddof=1)
int_lower_rgb = mean_rgb - t_29_0475*std_rgb/np.sqrt(30)
int_high_rgb = mean_rgb + t_29_0475*std_rgb/np.sqrt(30)
mean_shape = np.mean(shape_np)
std_shape = np.std(shape_np, ddof=1)
int_lower_shape = mean_shape - t_29_0475*std_shape/np.sqrt(30)
int_high_shape = mean_shape + t_29_0475*std_shape/np.sqrt(30)

print 'RGB'
print 'Mean', mean_rgb
print 'IC 95 [', int_lower_rgb, ',' , int_high_rgb, ']'

print 'SHAPE'
print 'Mean', mean_shape
print 'IC 95 [', int_lower_shape, ',' , int_high_shape, ']'