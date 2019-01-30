#!/usr/bin/env python
'''
Author Met
07/11/17 - 11:06
'''
from __future__ import division, print_function
import numpy as np
import h5py
import pickle
import logging
import scipy.io as sio
import os
log = logging.getLogger(__name__)


def import_vtc_data(VTC_path):
    ## Load VTC Data
    VTC_data_file = h5py.File(VTC_path, 'r')
    data_h5py = VTC_data_file.get('VTC')

    VTC_length = int(np.array(data_h5py.get('Length')))
    VTC_number = int(np.array(data_h5py.get('NumberVTC')))
    VTC_name = ''.join(chr(i) for i in np.array(data_h5py.get('name')))
    VTC_data = np.array(data_h5py.get('data')).T

    return VTC_length, VTC_name, VTC_number, VTC_data


def import_cnn_data(data_path, score_path, align_shift, VTC_length, selected_layer='fc7_R', transformation_type='none'):
    """
    Import CNN data from Faster-RCNN
    :param data_path: data path (both fc or conv)
    :param score_path: score path
    :param align_shift: align shift, MUST BE NEGATIVE
    :param VTC_length:
    :param selected_layer: between fc6, fc7, fc6_R, fc7_R or conv ...
    :return: scores, feature_zero_data
    """
    scores = None
    try:
        with open(score_path, 'rb') as sf:
            scores = pickle.load(sf)
            scores_list = []
            for i in range(np.array(scores.keys()).astype(np.int).max()):
                scores_list.append(scores[str(i)]['pred'])
            if align_shift>=0:
                scores = np.array(scores_list)[align_shift:][-VTC_length:]
            else:
                scores = np.array(scores_list)[:align_shift][-VTC_length:]
    except IOError:
        log.error("Score path not present: {}".format(score_path))

    try:
        with open(data_path, 'rb') as df:
            data = pickle.load(df)
    except IOError:
        log.error("Data path not present: {}".format(data_path))

    data_list = []
    for i in range(np.array(data.keys()).astype(np.int).max()):
        data_list.append(data[str(i)][selected_layer])

    features = np.vstack(data_list)  # (106, 4096) => time, features
    
    # Feature transformation
    features = feature_transformation(features, transformation_type)
    
    feature_zero_data = features
    feature_zero_data = features - features.mean(axis=0)
    feature_zero_data = feature_zero_data / (
                        feature_zero_data.std(axis=0) + (feature_zero_data.std(axis=0) == 0).astype(np.int))
    if align_shift>=0:
        feature_zero_data = feature_zero_data[align_shift:, :][-VTC_length:, :]
    else:
        feature_zero_data = feature_zero_data[:align_shift, :][-VTC_length:, :]

    return scores, feature_zero_data


def feature_transformation(X, transformation_type):
    
    X_transformed = X
    
    if transformation_type == 'none':
        X_transformed = X
    elif transformation_type == 'log':
        if np.min(X) < 0:
            log.error("ERROR 2.32: log transformation of negative values!")
        else:
            X_transformed = np.log(1 + X)
    elif transformation_type == 'L2norm':
        X_transformed = X / np.linalg.norm(X)
    elif transformation_type == 'exp':
        X_transformed = np.exp(X)
    elif transformation_type == 'square':
        X_transformed = np.square(X)
    elif transformation_type == 'sqrt':
        X_transformed = np.sqrt(X)
    else:
        log.error("ERROR 2.33: transformation type not recognised!")
    
    return X_transformed


def load_movies(Names_train, Vtc_mat_path_train, Cnn_data_path, score_path, GT_path, shift=-2, selected_layer='fc7_R', 
                transformation_type='none'):
    """
    Load VTCs of movies from a list of files and their associated CNN activation
    :param Names_train: list of mat files
    :param Vtc_mat_path_train: list of mat files per id
    :param Cnn_data_path:
    :param score_path:
    :param shift: time shift
    :param selected_layer:
    :return:
    """
    # Train data
    Vtc_length_train = []
    Vtc_names_train = []
    Vtc_number_train = []
    Vtc_data_train = []
    scores_movie_out_train = []
    feature_person_data_train = []
    GT = []

    for n_train_movie in range(len(Names_train)):
        # VTC data
        n_length, n_names, n_number, n_data = import_vtc_data(Vtc_mat_path_train[n_train_movie])
        Vtc_length_train.append(n_length)
        Vtc_names_train.append(n_names)
        Vtc_number_train.append(n_number)
        Vtc_data_train.append(n_data)

        # CNN data
        n_scores, n_features = import_cnn_data(Cnn_data_path[n_train_movie],
                                               score_path[n_train_movie],
                                               align_shift=shift,
                                               VTC_length=Vtc_length_train[n_train_movie],
                                               selected_layer=selected_layer,
                                               transformation_type=transformation_type)

        scores_movie_out_train.append(n_scores)
        feature_person_data_train.append(n_features)
        
        # GT data
        GT.append(load_gt_data(GT_path, n_names, n_number, shift, n_length))
        

    return Vtc_length_train, Vtc_names_train, Vtc_number_train, Vtc_data_train, \
           scores_movie_out_train, feature_person_data_train, GT


def load_gt_data(GT_fn, Names, num_VTC, align_shift, VTC_length):

    tmp = sio.loadmat(os.path.join(GT_fn, Names))
    tmp = tmp[tmp.keys()[0]]
    
    if align_shift>=0:
        GT = (np.tile(tmp[align_shift:][-VTC_length:], (num_VTC, 1)))     # repeat tmp for each subject
    else:
        GT = (np.tile(tmp[:align_shift][-VTC_length:], (num_VTC, 1)))     # repeat tmp for each subject
    return GT



#    if align_shift>=0:
#        GT = (np.tile(tmp[align_shift:][-VTC_length:], (num_VTC, 1)))     # repeat tmp for each subject
#    else:
#        GT = (np.tile(tmp[:align_shift][-VTC_length:], (num_VTC, 1)))     # repeat tmp for each subject





