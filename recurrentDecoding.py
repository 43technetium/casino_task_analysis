# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:41:06 2019

@author: Tomas G. Aquino

This file performs a general decoding analysis using recurrent neural networks 
described in drnn.py. After pre-processing and data selection, a network is 
trained and tested with cross-validation.

Neural data must be a 3d matrix of shape (n_trials, n_timepoints, n_features)
Target data must be a matrix of shape (n_trials, n_timepoints, 
n_reconstructed_targets).

The recurrent neural network implemented here is based on Haghi et al. (2019), 
NeurIPS. In the original work, wavelet features are constructed from LFP
for use with this decoder, but other types of neural features can be used. 
In this implementation, I chose single unit spikes. Additional feature 
construction steps can be added in the pre-processing section.

v2: incorporated structure from gbDecoding
"""

###############################################################################
# Libraries
###############################################################################

import drnn as drnn
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spi
import torch
import time
import os
import json

# For saving numpy arrays into JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

basefolder = r'C:/Users/Tomas/Documents/PhD/OLab/casinoTask/casinoTaskAnalysis/patientData/'
behavior_folder = basefolder+'allBehavior_intracranial/'
#file_name,model_name = 'smBI_qP_uI_rlAI_selected.mat', 'RL'
#file_name,model_name = 'smBI_qP_uI_rlActAI_selected.mat','Bayes'
#file_name,model_name = 'smBI_qP_nI_uI_rlAI_selected.mat','RL-nov'
file_name,model_name = 'smBI_qP_nID_uID_rlAI_scID_selected.mat','sticky-nov-unc'

behavior_file = behavior_folder+file_name
behavior_data = spi.loadmat(behavior_file)

#sessions = ['P41CS_100116','P43CS_110816','P47CS_021817','P51CS_070517','P56CS_042118','P56CS_042518','P60CS_100618','P61CS_022119','P62CS_041919','P63CS_082519']
sessions = ['P63CS_082519']
# Behavioral data is stored separately, and sessions require an unique ID to be correctly loaded from it
#session_file_id = [0,1,2,5,9,10,11,12,13,14]
session_file_id = [14]

areaNames = ['HIP','AMY','dACC','preSMA','vmPFC']

# Parameters for regression
# Multiple CV resamples
n_cv = 50
n_tested_parameters = 1

# Getting results from sessions
all_train_scores = np.zeros((len(sessions),n_cv,n_tested_parameters))
all_test_scores = np.zeros((len(sessions),n_cv,n_tested_parameters))
all_gb_features = []
all_regression_coefficients = []
all_brain_areas = []
optimal_unit_parameters = []
all_rsquared = np.zeros(len(sessions))
        
# Loading each session separately
for sI in np.arange(len(sessions)):
    print(sessions[sI])
    datafolder = basefolder+sessions[sI]+'/'
    filename = 'sessionData.mat'
    mat_session = spi.loadmat(datafolder+filename)
    sessionData = mat_session['sessionData']
    trialStartTime = np.squeeze(sessionData[0,0]['trialStartTime'])
    trialResponseTime = np.squeeze(sessionData[0,0]['trialResponseTime'])
    trialOutcomeTime = np.squeeze(sessionData[0,0]['trialOutcomeTime'])
    trialEndTime = np.squeeze(sessionData[0,0]['trialEndTime'])
    
    # all_rsquared[sI] = np.squeeze(behavior_data['fitResults'][0,0]['latents'][0,session_file_id[sI]]['Rsquared'])
    
    # Time window configuration
    # 50 ms windows
    windowSize = 0.05   
    #windowSize = 1
    
    # For post reward
    windowStarts = np.arange(0,2,windowSize)
    # For pre response
    # windowStarts = np.arange(-2,0,windowSize)
    # Pre response, normalized time axis
    # windowStarts = np.arange(0,1,windowSize)
    
    windowEnds = windowStarts+windowSize
    
    # 500 ms windows
    # windowStarts = [0,0.5,1,1.5]
    # windowEnds = [0.5,1.5,1.5,2]
    # Whole period
    #windowStarts = [0]
    #windowEnds = [2]
    
    # Looping over units
    nUnits = mat_session['sessionData'][0,0]['neuralData'][0,0]['unitCell'].size
    brainAreas = mat_session['sessionData'][0,0]['neuralData'][0,0]['mergedAreas']
    if len(all_brain_areas)==0:
        all_brain_areas = brainAreas
    else:
        all_brain_areas = np.vstack((all_brain_areas,brainAreas))
    nTrials = len(trialOutcomeTime)
    # Defining spike tensors
    postOutcomeSpikes = np.zeros((nTrials,len(windowStarts),nUnits))
    preDecisionSpikes = np.zeros((nTrials,len(windowStarts),nUnits))
    preDecisionSpikes_normalized = np.zeros((nTrials,len(windowStarts),nUnits))
    trialReferencedSpikes = np.zeros((nTrials,len(windowStarts),nUnits))
    all_preResponseSpikes = np.zeros((nTrials,nUnits))
    for uI in np.arange(nUnits):
        unitData = mat_session['sessionData'][0,0]['neuralData'][0,0]['unitCell'][uI]    
        for tI in np.arange(nTrials):
            trialSpikes = np.squeeze(unitData[0]['trialReferencedSpikes'][0,0][0,tI])
            trialSpikes_decision = np.squeeze(unitData[0]['decisionReferencedSpikes'][0,0][0,tI])
            trialSpikes_outcome = np.squeeze(unitData[0]['outcomeReferencedSpikes'][0,0][0,tI])
            all_preResponseSpikes[tI,uI] = np.sum(trialSpikes<trialResponseTime[tI])
            for timeI in np.arange(len(windowStarts)):            
                trialReferencedSpikes[tI,timeI,uI] = np.sum(np.logical_and(trialSpikes>windowStarts[timeI], trialSpikes<windowEnds[timeI]))
                postOutcomeSpikes[tI,timeI,uI] = np.sum(np.logical_and(trialSpikes_outcome>windowStarts[timeI], trialSpikes_outcome<windowEnds[timeI]))
                preDecisionSpikes[tI,timeI,uI] = np.sum(np.logical_and(trialSpikes_decision>windowStarts[timeI], trialSpikes_decision<windowEnds[timeI]))
                preDecisionSpikes_normalized[tI,timeI,uI] = np.sum(np.logical_and(trialSpikes/trialResponseTime[tI]>windowStarts[timeI], trialSpikes/trialResponseTime[tI]<windowEnds[timeI]))

    # Getting variable to decode    
    # Categorical regressors
    regressor_name, mode = 'outcome', 'classification'
    # regressor_name, isClass = 'decisionCategory', True
    # regressor_name, isClass = 'respKey', True
    
    # Continuous regressors
    # regressor_name, isClass = 'select_qVals', False
    # regressor_name, isClass = 'pChoice', False
    # regressor_name, isClass = 'diff_qVals', False
    # regressor_name, isClass = 'diff_stimUtil', False
    # regressor_name, isClass = 'reject_qVals', False
    # regressor_name, isClass = 'select_stimUtil', False
    # regressor_name, isClass = 'reject_stimUtil', False
    # regressor_name, isClass = 'select_rlActVal', False
    # regressor_name, isClass = 'select_uUtil', False
    # regressor_name, isClass = 'diff_uUtil', False
    # regressor_name, isClass = 'reject_uUtil', False
    # regressor_name, isClass = 'RPE', False
    # regressor_name, isClass = 'outcome', False
    # regressor_name, isClass = 'select_nVal', False
    
    
    # Setting up training and testing folds
    # X = trialReferencedSpikes
    # X = preDecisionSpikes
    X = postOutcomeSpikes
    # X = all_preResponseSpikes/trialResponseTime[:,None]
    # X = preDecisionSpikes_normalized    
    
    regressor = np.squeeze(behavior_data['fitResults'][0,0]['latents'][0,session_file_id[sI]][regressor_name])
    regressor[np.isnan(regressor)] = 0    
    #regressor = stats.zscore(regressor)
    
    # Clearing nans
    regressor[np.isnan(regressor)] = 0
    # Adding repetitions over time to outcome vector
    regressor = regressor.reshape(len(regressor),1)
    n_times = X.shape[1]
    regressor = np.tile(regressor,n_times)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, regressor, test_size=0.2)    

    if mode == 'classification':
        n_classes = 2
        n_outputs = n_classes
    elif mode == 'regression':
        n_outputs = 1
    
    training_params = {'n_features':X.shape[2],'batch_size':1,'n_nodes':10,
                       'n_outputs':n_outputs,'dropout_coef':0.25,'mode':mode}    
    model = drnn.drnn_decoder(**training_params)
    
    
    # Saving analysis params
    analysis_params = {'training_params': training_params, 'X_train': X_train,
                       'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
                       'n_times': n_times, 'regressor_name': regressor_name,
                       'mode': mode, 'nUnits': nUnits, 'brainAreas': brainAreas,
                       'nTrials': nTrials, 'windowStarts': windowStarts,
                       'windowEnds': windowEnds, 'windowSize': windowSize,
                       'sessionFile': datafolder+filename, 'session': sessions[sI],
                       'behavior_file': file_name, 'model_name': model_name}    
    
    # Saving analysis params
    save_folder = datafolder + 'saved_drnn_models/'
    if os.path.isdir(save_folder) == False:
        os.mkdir(save_folder)
    timestamp = str(int(time.time()))
    params_file = save_folder + sessions[sI] + '_' + timestamp + '_analysis_params.json'
    with open(params_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_params, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    
    
    # Loading previously trained model
    '''    
    model = torch.load(save_folder + 'P63CS_082519_1580109478_checkpoint.pt')
    # Loading JSON file
    with open(save_folder + 'P63CS_082519_1580109478_analysis_params.json') as json_file:
        data = json.load(json_file)
    X_train = np.array(data['X_train'])
    y_train = np.array(data['y_train'])
    X_test = np.array(data['X_test'])
    '''
    
    # The model's mode can be classification or regression
    model.fit(X_train,y_train)                  
    # Saving PyTorch model
    torch.save(model, save_folder + sessions[sI] + '_' + timestamp + '_checkpoint.pt')
    
    y_hat_train = model.predict(X_train,y_train)
    y_hat_test = model.predict(X_test,y_test)
    
    
    
    '''

    if isClass: # Run gradient boosting classification for discrete variables  
        n_max_estimators = 300
        test_deviance = np.zeros((n_cv,n_max_estimators))
        feature_importances = np.zeros((n_cv,X.shape[1]))      
        # Looping over CV folds
        for cI in np.arange(n_cv):
            print(cI)            
            X_train, X_test, y_train, y_test = train_test_split(X, regressor, test_size=0.2)
            # Getting optimal estimator value (hyperparameter)
            X_train_train,X_train_test,y_train_train,y_train_test = train_test_split(X_train, y_train, test_size=0.2)                        
            '''



'''
###############################################################################
# Loading neural data
###############################################################################

data = r"C:/Users/Tomas/Documents/PhD/OLab/casinoTask/casinoTaskAnalysis/patientData/tensors/cumulativeSpikeTensor_sparse.mat"
x = loadmat(data)

# The spikes tensor is organized in (units,trials,timeBins)
spikes = x['data']['spikeTensor'][0][0]
rewards = x['data']['rewards'][0][0]
sessionID = x['data']['sessionID'][0][0]
n_units_per_session = np.unique(sessionID,return_counts=True)[1]
trial_sessionID = x['data']['trial_sessionID'][0][0]
mean_final_counts = x['data']['mean_final_counts'][0][0]


###############################################################################
# Pre-processing and data selection
###############################################################################

# Perform pre-processing (e.g. PCA, normalization, shifting) and data selection
# (e.g. select brain areas) here and create matrices X and y

selected_session = 8
X_neuron = spikes[np.squeeze(sessionID==selected_session),:,:]
X = X_neuron[:,np.squeeze(trial_sessionID==selected_session),:]
y = rewards[trial_sessionID==selected_session]
# Clearing nans
y[np.isnan(y)] = 0

# Enforcing correct axes on data
X = np.swapaxes(X,0,1)
X = np.swapaxes(X,1,2)

# Adding repetitions over time to outcome vector
y = y.reshape(len(y),1)
n_times = X.shape[1]
y = np.tile(y,n_times)

###############################################################################
# Cross-validation training/testing
###############################################################################
# Can be classification or regression
mode = 'classification'

if mode == 'classification':
    n_classes = 2
    n_outputs = n_classes
elif mode == 'regression':
    n_outputs = 1

training_params = {'n_features':X.shape[2],'batch_size':1,'n_nodes':10,
                   'n_outputs':n_outputs,'dropout_coef':0.25,'mode':mode}

# Creating cross-validation train-test split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

model = drnn.drnn_decoder(**training_params)
# The model's mode can be classification or regression
model.fit(X_train,y_train)
y_hat_test = model.predict(X_test)


###############################################################################
# Plotting some results
###############################################################################
plt.figure()
plt.plot(y_test,y_hat_test.cpu().detach().numpy(),'bo')
plt.show()
'''



