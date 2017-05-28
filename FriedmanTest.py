# By Felipe {fnba}, Paulo {pan2} e Debora {dca2}@cin.ufpe.br
#Chi_square table used
# https://www.medcalc.org/manual/chi-square-table.php

import pickle
import pandas as pd
import numpy as np
import warnings

# ignore warnings
warnings.filterwarnings('ignore')


class FriedmanTest:

    def __init__(self, bayes_result=None, knn_result=None, majority_result=None, all_results=None):
        self.data_frame = None
        self.all_accuracies = None
        self.bayes_result = bayes_result
        self.knn_result = knn_result
        self.majority_result = majority_result
        self.n_rows = None
        self.all_results = all_results

        if bayes_result is None:
            raise ValueError('No result for Bayes Classifier defined')
        if knn_result is None:
            raise ValueError('No result for KNN Classifier defined')
        if majority_result is None:
            raise ValueError('No result for Majority Vote Classifier defined')
        if all_results is None:
            raise ValueError('No Array Results were found')

        self.read_from_csv('chi_square.csv')

    def read_from_csv(self, filename):
        print 'Reading Chi-Square table...'
        self.data_frame = pd.read_csv(filename, sep=",", header=0)

    def get_from_pickle_pandas(self, filename):
        self.all_accuracies = pd.read_pickle(filename)

    @staticmethod
    def get_xr_pow2(rank_tmp, n, k):
        # n = number_of_rows
        n = n

        # k = number_of_columns
        k = k

        RjPow2 = 0
        xr_pow_2 = 0

        # Rj = sum of the results per columns,
        # in this case we've only one value for each group/column
        value = np.sum(rank_tmp, axis=0)
        RjPow2 += pow(value, 2)

        # xr_pow_2 = (12/n * k*(k+1)*[RjPow2-3*n(k+1)]
        calc_part_1 = (n * k) * (k + 1)
        calc_part_1 = 12.0 / calc_part_1
        calc_part_2 = (3 * n) * (k + 1)
        calc_part_2 = RjPow2 - calc_part_2
        xr_pow_2 = calc_part_1 * calc_part_2

        if xr_pow_2 == 0:
            xr_pow_2 = 0.000000000001

        xr_pow_2 = xr_pow_2/100

        return xr_pow_2

    def convert_to_decimal(self, value):
        if value == 0:
            value = 0.00000000001
        else:
            value = value / 100

        return value

    def evaluate_classifiers(self, n_rows, x_pow_2):
        df = self.data_frame
        arr = np.array(df)
        if n_rows == 1:
            print 'Defining search for the default value of N'
            n_rows -= 1

        #aproves if the classifiers have used the same data source
        row_counter = 0
        for row in arr:
            if row_counter == n_rows:
                max_value = np.max(row)
                min_value = np.min(row)
                if min_value <= x_pow_2 <= max_value:
                    is_approved = True
                    return is_approved
            row_counter += 1


    def friedman_test(self):
        bayes = self.bayes_result
        knn = self.knn_result
        majority = self.majority_result

        classifiers = ['Bayes', 'Knn', 'Majority']
        table_of_values = self.all_results
        results = self.all_results

        rank_tmp = np.zeros(shape=(table_of_values.__len__(), table_of_values[0].__len__()))
        value_found = False
        for g in table_of_values:
            max_value = np.max(g)
            min_value = np.min(g)

            for value in g: #for each value in group...
                value_found = False
                l = rank_tmp.__len__()
                l_a = rank_tmp[0].__len__()

                # k = loops over lines
                for k in xrange(rank_tmp.__len__()):
                    if value_found is True:
                        break

                    # c - loops over columns
                    for c in xrange(rank_tmp[0].__len__()):
                        if value == max_value:
                            if rank_tmp[k][c] == 0:
                                rank_tmp[k][c] = classifiers.__len__()
                                value_found = True
                                break
                        if value == min_value:
                            if rank_tmp[k][c] == 0:
                                rank_tmp[k][c] = classifiers.__len__() - 2
                                value_found = True
                                break
                        else:
                            if rank_tmp[k][c] == 0:
                                rank_tmp[k][c] = classifiers.__len__() - 1
                                value_found = True
                                break

        sum_weights = 0

        sum_all_weights = np.sum(rank_tmp, axis=0)

        n = rank_tmp.__len__()
        k = rank_tmp[0].__len__()

        x_r_pow_2 = self.get_xr_pow2(sum_all_weights, n, k)

        #get number of rows (groups)
        n = rank_tmp.__len__()

        is_approved = self.evaluate_classifiers(n, x_r_pow_2)

        print 'Defining Null Hyphotesis...'
        print 'H_0: The classifiers are equivalent'
        print 'H_1: The classifiers are not equivalent'
        print 'The chi-square from xr_pow_2 = (12/n * k*(k+1)*[RjPow2-3*n(k+1)]  is ', x_r_pow_2
        print 'The number of groups is ', n
        print 'The number of tests is ', classifiers.__len__()
        print 'For n = ', n, 'then H_0 is acceptance is ', is_approved

        print 'Done'

# ---------------

# Begin
arr = np.zeros(shape=(2, 3))
arr[0][0]=0.79
arr[0][1]=0.89
arr[0][2]=0.88
arr[1][0]=0.55
arr[1][1]=0.80
arr[1][2]=0.81
ft = FriedmanTest(bayes_result=0.79, knn_result=0.89, majority_result=0.88, all_results=arr)
ft.friedman_test()