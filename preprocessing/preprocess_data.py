#!/usr/bin/env python
'''
Author Met
07/11/17 - 11:06
'''
from __future__ import division, print_function
import logging
import numpy as np
from sklearn.preprocessing import Imputer
log = logging.getLogger(__name__)

def preprocessing_nan_value(file_data):
    # Imputation of the missing values: mean
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(file_data)
    file_data = imp.transform(file_data)

    return file_data


def train_data_preparation(Vtc_data, feature_person_data, GT, Vtc_length, Vtc_number, Names, Ratio_subjects_to_keep=0.5):
    """
        Data preparation for training-set
        :param Vtc_data: input VTC data from @load_movies
        :param feature_person_data: activation from @load_movies
        :param Vtc_length: length of each movie
        :param Vtc_number: NumberVTC for each movie
        :param Names: Movies names
        :param Ratio_subjects_to_keep: sub-sampling
        :return: cca_X, cca_Y
    """
    cca_Y = []
    cca_X = []
    GT_labels = []
    
    for n_movie in range(len(Names)):
        # Full movie
        cca_Y_n_movie = preprocessing_nan_value(Vtc_data[n_movie])
        cca_X_n_movie = np.tile(feature_person_data[n_movie], (Vtc_number[n_movie],1))
        
        # Select part of it, just doing a random sampling
        indices_to_keep = np.random.permutation(Vtc_number[n_movie] * Vtc_length[n_movie])
        indices_to_keep = indices_to_keep[:int(indices_to_keep.shape[0]*Ratio_subjects_to_keep)]
        
        cca_Y_n_movie = cca_Y_n_movie[indices_to_keep,:]
        cca_Y.append(cca_Y_n_movie)

        cca_X_n_movie = cca_X_n_movie[indices_to_keep,:]
        cca_X.append(cca_X_n_movie)
        
        GT_labels.append(GT[n_movie][indices_to_keep,:])

    return cca_X, cca_Y, GT_labels


def test_valid_preparation(Vtc_data, feature_person_data, Vtc_length, Vtc_number, Names):
    """
    Data preparation for validation/test-set
    :param Vtc_data: input VTC data from @load_movies
    :param feature_person_data: activation from @load_movies
    :param Vtc_length: length of each movie
    :param Vtc_number: NumberVTC for each movie
    :param Names: Movies names
    :return:
    """
    cca_Y = []
    cca_X = []
    testing_num_VTC_test = []
    for n_test_movie in range(len(Names)):
        cca_Y.append( preprocessing_nan_value(Vtc_data[n_test_movie]) )

        n_VTC_to_test = Vtc_number[n_test_movie]
        testing_num_VTC_test.append( n_VTC_to_test )
        cca_X.append( np.tile(feature_person_data[n_test_movie], (n_VTC_to_test,1)) )

    return cca_X, cca_Y, testing_num_VTC_test










