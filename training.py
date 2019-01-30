# -*- coding: utf-8 -*-
"""
Michele Svanera, November 2017

Reduced Rank Regression

The code does:
* take in input movie VTCs and fc7 description
* perform a grid search parameter estimation for
* rank
* regularisation
* time_shift

"""


import numpy as np
import sys
from scipy import stats
import scipy.io as sio
import argparse
from sklearn.metrics import r2_score
from sklearn import linear_model

import reduced_rank_regressor as RRR

import os
# Optimization Baesian search
from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize, forest_minimize, dummy_minimize
from skopt.plots import plot_convergence
#from skopt import dump, load
#from commentjson import dump
from functools import partial
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
from preprocessing.import_data import *
from preprocessing.preprocess_data import *
from utility.utils import *
from utility.multi_logger import initialzie_logger
import traceback
import itertools

########################################## Path settings + Constant ##########################################
dir_save = './results_RRRR/'        # where to save the results
save_dump_flag = True           # Save corr results
save_fig_flag = False            # Save figure

VTC_path = './data/all_voxels/'
Fc7_path = './data/cnn/VGG16_person/'
GT_fn = './data/GT_faceFullBody/3_types/'

Names_train = ['Sophie_f.mat', 'Avenge.mat', 'Ring.mat', \
               'Stepmom_f.mat', 'X-files.mat']
Names_train_fc = ['The_Sophie_fc.pickle', 'Avenge_4_cut_fc.pickle', \
                  'Ring_cut_fc.pickle', 'Stepmom_gain2_cut_fc.pickle', 'Xfiles_distress_fc.pickle']
Names_train_pr = ['The_Sophie_preds.pickle', 'Avenge_4_cut_preds.pickle', \
                  'Ring_cut_preds.pickle', 'Stepmom_gain2_cut_preds.pickle', 'Xfiles_distress_preds.pickle']

Names_valid = ['Seven.mat', 'Shining.mat', 'Mary.mat']
Names_valid_fc = ['Seven_cut_fc.pickle', 'Shining_cut_fc.pickle', 'SomethingAboutMary_cut_fc.pickle']
Names_valid_pr = ['Seven_cut_preds.pickle', 'Shining_cut_preds.pickle', 'SomethingAboutMary_cut_preds.pickle']

Names_test = ['Poet.mat', 'Denali.mat', 'Forrest.mat', 'Ryan.mat', \
              'Black_Swan.mat']
Names_test_fc = ['Dead_Poet_cut_fc.pickle', 'Denali_fc.pickle', 'ForrestGump_fc.pickle', \
                 'Private_Ryan_cut_fc.pickle', 'Black_Swan_cut_fc.pickle']
Names_test_pr = ['Dead_Poet_cut_preds.pickle', 'Denali_preds.pickle', 'ForrestGump_preds.pickle', \
                 'Private_Ryan_cut_preds.pickle', 'Black_Swan_cut_preds.pickle']

Vtc_mat_path_train = [VTC_path + name for name in Names_train]
Cnn_mat_path_train = [Fc7_path + name for name in Names_train_fc]
Score_path_train = [Fc7_path + name for name in Names_train_pr]

Vtc_mat_path_valid = [VTC_path + name for name in Names_valid]
Cnn_mat_path_valid = [Fc7_path + name for name in Names_valid_fc]
Score_path_valid = [Fc7_path + name for name in Names_valid_pr]

Vtc_mat_path_test = [VTC_path + name for name in Names_test]
Cnn_mat_path_test = [Fc7_path + name for name in Names_test_fc]
Score_path_test = [Fc7_path + name for name in Names_test_pr]


def objective(params, X_train, Y_train, X_valid,Y_valid,
              correlation_measure='pearsonr',shift=''):
    assert correlation_measure == 'pearsonr' or correlation_measure == 'r2' or correlation_measure == 'MSE'
    rank, reg = params
    log.debug("Objective: {}".format(params))
    
    # Create a RRRR object as an instantiation of the RRRR object class
    rrr = RRR.ReducedRankRegressor(rank=rank, reg=reg)

    # Use the fit() method to find a RRRR mapping
    log_train = rrr.fit(np.concatenate(Y_train), np.concatenate(X_train))
    log.debug("Training: {}".format(log_train))
    
    ########################################## Testing ##########################################
    
    log.debug("Testing: {}".format(params))
    
    ###################### Training set ######################
    log.info("*** Corr on Training set ***")

    corr_values = []                # NO Cicle over every subject. Just one shot (per movie)
    for n_movie in range(len(Names_train)):
        Y_movie_n = Y_train[n_movie]
        X_movie_n_hat = rrr.predict(Y_movie_n).real
        X_movie_n = X_train[n_movie]

        if correlation_measure == 'pearsonr':
            corr_value, _ = stats.pearsonr(np.concatenate(X_movie_n_hat),np.concatenate(X_movie_n))
        elif correlation_measure == 'MSE':
            corr_value = -(np.power((X_movie_n_hat - X_movie_n),2)/np.prod(X_movie_n.shape)).mean()
        else:
            corr_value = r2_score(np.concatenate(X_movie_n_hat),np.concatenate(X_movie_n))
    
        corr_values.append(corr_value)

    log_print = ''.join([' '+title[:-4]+'=' + '%.3f'%(corr) for title,corr in itertools.izip_longest(Names_train,corr_values)])
    log.info('Training Movies:' +  "{}".format(log_print))


#    log.info("*** Corr on Training set (subject-based) ***")
#    for n_movie in range(len(Names_train)):     ## For every movies
#        corr_values = []
#        for n_subject in range(Vtc_number_train[n_movie]):      ## For every subject
#            #Select different subject and find correlation
#            begin_test_index = n_subject * Vtc_length_train[n_movie]
#            end_test_index  = (n_subject+1) * Vtc_length_train[n_movie]
#
#            Y_movie_n = Y_train[n_movie][begin_test_index:end_test_index,:]
#            X_movie_n_hat = rrr.predict(Y_movie_n).real
#            X_movie_n = X_train[n_movie][begin_test_index:end_test_index,:]
#
#            if correlation_measure == 'pearsonr':
#                corr_value, _ = stats.pearsonr(np.concatenate(X_movie_n_hat),np.concatenate(X_movie_n))
#            elif correlation_measure == 'MSE':
#                corr_value = -(np.power((X_movie_n_hat - X_movie_n),2)/np.prod(X_movie_n.shape)).mean()
#            else:
#                corr_value = r2_score(np.concatenate(X_movie_n_hat),np.concatenate(X_movie_n))
#
#            corr_values.append(corr_value)
#
#        log_print = ''.join([Names_train[n_movie][:-4]+' --> '] + ['mean: ' +'%.6f'%(np.mean(corr_values))+
#                            ' - var: ' +'%.6f'%(np.var(corr_values)) + ' - std: ' +'%.6f'%(np.std(corr_values)) + ' - ALL: '] +
#                            [' %.6f'%(corr) for corr in corr_values])
#        log.info('(Training) Movie:' +  "{}".format(log_print))


    ###################### Validation set ######################
    log.info("*** Corr on Validation set ***")
    
    corr_values = []                # NO Cicle over every subject. Just one shot (per movie)
    for n_movie in range(len(Names_valid)):
        Y_movie_n = Y_valid[n_movie]
        X_movie_n_hat = rrr.predict(Y_movie_n).real
        X_movie_n = X_valid[n_movie]
        
        if correlation_measure == 'pearsonr':
            corr_value, _ = stats.pearsonr(np.concatenate(X_movie_n_hat),np.concatenate(X_movie_n))
        elif correlation_measure == 'MSE':
            corr_value = -(np.power((X_movie_n_hat - X_movie_n),2)/np.prod(X_movie_n.shape)).mean()
        elif correlation_measure == 'r2':
            corr_value = r2_score(np.concatenate(X_movie_n_hat),np.concatenate(X_movie_n))
        else:
            raise NameError('Error 1.7! Corr function not understood!')
        corr_values.append(corr_value)

    log_print = ''.join([' '+title[:-4]+'=' + '%.3f'%(corr) for title,corr in itertools.izip_longest(Names_valid,corr_values)])
    log.info('Validation Movies:' +  "{}".format(log_print))

    mean_across_films = np.mean(corr_values)

#    log.info("*** Corr on Validation set (subject-based) ***")
#    for n_movie in range(len(Names_valid)):     ## For every movies
#        corr_values = []
#        for n_subject in range(Vtc_number_valid[n_movie]):      ## For every subject
#            #Select different subject and find correlation
#            begin_test_index = n_subject * Vtc_length_valid[n_movie]
#            end_test_index  = (n_subject+1) * Vtc_length_valid[n_movie]
#
#            Y_movie_n = Y_valid[n_movie][begin_test_index:end_test_index,:]
#            X_movie_n_hat = rrr.predict(Y_movie_n).real
#            X_movie_n = X_valid[n_movie][begin_test_index:end_test_index,:]
#
#            if correlation_measure == 'pearsonr':
#                corr_value, _ = stats.pearsonr(np.concatenate(X_movie_n_hat),np.concatenate(X_movie_n))
#            elif correlation_measure == 'MSE':
#                corr_value = -(np.power((X_movie_n_hat - X_movie_n),2)/np.prod(X_movie_n.shape)).mean()
#            else:
#                corr_value = r2_score(np.concatenate(X_movie_n_hat),np.concatenate(X_movie_n))
#
#            corr_values.append(corr_value)
#
#        log_print = ''.join([Names_valid[n_movie][:-4]+' --> '] + ['mean: ' +'%.6f'%(np.mean(corr_values))+
#                            ' - var: ' +'%.6f'%(np.var(corr_values)) + ' - std: ' +'%.6f'%(np.std(corr_values)) + ' - ALL: '] +
#                            [' %.6f'%(corr) for corr in corr_values])
#        log.info('(Validation) Movie:' +  "{}".format(log_print))

    # Use the validate() method to test how well the CCA mapping generalizes to the test data.
    # For each dimension in the test data, correlations between predicted and actual data are computed.
    log.debug("ValidSet corr mean across movies {}\n".format(mean_across_films))
    return -mean_across_films


def main():
    log.info("Start\n\n")
    for shift in shifts:
        ########################################## Load Data ##########################################
        log.info("******************************************************************************\n\n")
        log.info("Shift {}, Loading movies...".format(shift))

        Vtc_length_train, Vtc_names_train, Vtc_number_train, \
        Vtc_data_train, _, feature_person_data_train, GT_train = load_movies(Names_train,
                                                                   Vtc_mat_path_train,
                                                                   Cnn_mat_path_train,
                                                                   Score_path_train,
                                                                   GT_path=GT_fn,
                                                                   shift=shift,
                                                                   selected_layer=selected_layer,
                                                                   transformation_type=feature_transform)
        log.debug("Vtc_length_train: {}\nVtc_names_train: {}\nVtc_number_train: {}\n".format(Vtc_length_train,
                                                                                 Vtc_names_train,
                                                                                 Vtc_number_train))
        Vtc_length_valid, Vtc_names_valid, Vtc_number_valid, \
        Vtc_data_valid, _, feature_person_data_valid, GT_valid = load_movies(Names_valid,
                                                                   Vtc_mat_path_valid,
                                                                   Cnn_mat_path_valid,
                                                                   Score_path_valid,
                                                                   GT_path=GT_fn,
                                                                   shift=shift,
                                                                   selected_layer=selected_layer,
                                                                   transformation_type=feature_transform)
        log.debug("Vtc_length_valid: {}\nVtc_names_valid: {}\nVtc_number_valid: {}\n".format(Vtc_length_valid,
                                                                                 Vtc_names_valid,
                                                                                 Vtc_number_valid))

        ########################################## Data preparation ##########################################
        X_train, Y_train, GT_train = train_data_preparation(Vtc_data_train,
                                                          feature_person_data_train,
                                                          GT_train,
                                                          Vtc_length_train,
                                                          Vtc_number_train,
                                                          Names_train,
                                                          Ratio_subjects_to_keep)

        X_valid, Y_valid, valid_num_VTC_test = test_valid_preparation(Vtc_data_valid,
                                                                      feature_person_data_valid,
                                                                      Vtc_length_valid,
                                                                      Vtc_number_valid,
                                                                      Names_valid)

        log.info("Loading and preparation process done")
        del (Vtc_data_train)
        del (Vtc_data_valid)
        ########################################## RRRR ##########################################
        log.info("\n\n[ RRRR Training Start ]")

        ###################### Training RRRR ######################
        space = [Integer(1, 50),
                 Real(1, 1e+12, "log-uniform")]

        log.info("Start optimiser.. ")
        func = partial(objective,
                       X_train=X_train,
                       Y_train=Y_train,
                       X_valid=X_valid,
                       Y_valid=Y_valid,
                       correlation_measure=correlation_measure,
                       shift=shift)

        # gp_minimize, forest_minimize, dummy_minimize
        res_gp = forest_minimize(func, space,
                                 n_calls=n_calls,
                                 n_random_starts=n_random_starts,
                                 random_state=random_state,
                                 base_estimator="RF",
                                 verbose=True,
                                 n_jobs=-1)

        log.debug("Test done. Res {}".format(res_gp))

        def do_test(kern, res_gp, numCC=10):
            ## Test on testing data ##
            return 0

        try:
            out = do_test(1,2,3)
            log.info(">> Do_test result: {}".format(out))
        except Exception as e:
            log.critical('Critical Event Notification after test\n\nTraceback:\n %s\n Errors: %s\n res_gp: %s',
                         (''.join(traceback.format_stack(), e, res_gp)))

        log.info("Test done. Mean across films {}\n".format(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fMRI Project. Train a RRRR using fMRI and CNN. Do a parameter search trough a bayesian optimization method.')
    parser.add_argument('--shifts', nargs='+', type=int, default=-2, help='shifts (one or more)')
    parser.add_argument('--Ratio_subjects_to_keep', type=float, default=1.0, help='subsample dataset')
    parser.add_argument('--n_calls', type=int, default=25, help='number of stemps made by the optimizer')
    parser.add_argument('--n_random_starts', type=int, default=20,
                        help='number of initial random steps made by the optimizer')
    parser.add_argument('--random_state', type=int, default=123, help='random seed')
    parser.add_argument('--correlation_measure', type=str, default='pearsonr',
                        help='correlation_measure: pearsonr or r2')
    parser.add_argument('--selected_layer', type=str, default='fc7_R', help='which cnn layer link')
    parser.add_argument('--log_name', nargs='+', type=str, default="debug.log", help='debug_filename')
    parser.add_argument('--feature_transform', type=str, default="none", help='transform to CNN features: none, log, L2norm, exp, square, sqrt')
    
    args = parser.parse_args()
    if type(args.shifts) != list:
        shifts = [args.shifts]
    else:
        shifts = args.shifts
    Ratio_subjects_to_keep = args.Ratio_subjects_to_keep
    n_calls = args.n_calls
    n_random_starts = args.n_random_starts
    random_state = args.random_state
    correlation_measure = args.correlation_measure
    selected_layer = args.selected_layer
    feature_transform = args.feature_transform

    log = initialzie_logger(log_name=str(args.log_name[0]))
    
    log.info("####### SELECTED PARAMETERS #######")
    for arg in vars(args):
        log.info((arg, getattr(args, arg)))
    log.info("#" * 50)

    try:
        main()
    except Exception as e:
         log.critical('Critical Event Notification\n\n Errors: {} Traceback {}'.format(''.join(traceback.format_stack()), e))


    log.info("DONE!")
    log.critical("END ANALYSIS")
