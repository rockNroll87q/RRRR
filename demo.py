#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Michele Svanera
"""

########################################## Import ##########################################
    
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

import reduced_rank_regressor as RRR


######################################### Path settings + Constant ##########################################

N_PARAMETERS_GRID_SEARCH = 20 
Data_path = "../data/"


########################################## Load Data ##########################################

trainX = np.loadtxt(Data_path+"trainX.txt")
testX = np.loadtxt(Data_path+"testX.txt")
validX = np.loadtxt(Data_path+"validX.txt")

trainY = np.loadtxt(Data_path+"trainY.txt")
testY = np.loadtxt(Data_path+"testY.txt")
validY = np.loadtxt(Data_path+"validY.txt")


########################################## Training ##########################################

# Cross-validation setup. Define search spaces
rank_grid               = np.linspace(1,min(min(trainX.shape),min(trainY.shape)), num=N_PARAMETERS_GRID_SEARCH)
rank_grid               = rank_grid.astype(int)
reg_grid                = np.power(10,np.linspace(-20,20, num=N_PARAMETERS_GRID_SEARCH+1))
parameters_grid_search  = {'reg':reg_grid, 'rank':rank_grid}

valid_test_fold         = np.concatenate((np.zeros((trainX.shape[0],))-1,np.zeros((validX.shape[0],))))
ps_for_valid            = PredefinedSplit(test_fold=valid_test_fold)

# Model initialisation
rrr                     = RRR.ReducedRankRegressor()#rank, reg)
grid_search             = GridSearchCV(rrr, parameters_grid_search, cv=ps_for_valid,
                                       scoring='neg_mean_squared_error', n_jobs = -1)

grid_search.fit(np.concatenate((trainX,validX)), np.concatenate((trainY,validY)))

# takes ~10 min on a regular desktop pc

# Display the best combination of values found
grid_search.best_params_
means = grid_search.cv_results_['mean_test_score']
means = np.array(means).reshape(N_PARAMETERS_GRID_SEARCH, N_PARAMETERS_GRID_SEARCH+1)
grid_search.best_score_

# Graph
scores = [x[1] for x in grid_search.grid_scores_]
scores = np.array(scores).reshape(N_PARAMETERS_GRID_SEARCH, N_PARAMETERS_GRID_SEARCH+1)

import plotly
import plotly.graph_objs as go

data = [go.Surface(z=scores)]   #means
layout = go.Layout(
                    scene = dict(
                    xaxis = dict(
                        title='Regression',
                        ticktext= [str("%0.*e"%(0,x)) for x in reg_grid][::2],
                        tickvals= range(reg_grid.shape[0])[::2]
                        ),
                    yaxis = dict(
                        title='Rank',
                        ticktext= [str("%0.*e"%(0,x)) for x in rank_grid][::2],
                        tickvals= range(rank_grid.shape[0])[::2]
                        ),
                    zaxis = dict(
                        title='Train error'),
                    )
                    )
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='elevations-3d-surface')



# Train a model with the best set of hyper-parameters found
rrr.rank                = int(grid_search.best_params_['rank'])
rrr.reg                 = grid_search.best_params_['reg']
rrr.fit(trainX, trainY)


########################################## Testing ##########################################

Yhat                    = rrr.predict(testX).real

MSE                     = (np.power((testY - Yhat),2)/np.prod(testY.shape)).mean()
print MSE

diag_corr               = (np.diag(np.corrcoef(testY,Yhat)))
print diag_corr.mean()

    
    
    
    
    
    
    
    
