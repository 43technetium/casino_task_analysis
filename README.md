# casino_task_analysis
Custom analysis scripts for the Explore-exploit/Casino Task in Aquino et al. (2023), Nature Human Behavior.

## behaviorSummaryPlot.m
This function takes in behavioral data and generate the following summaries (relevant for Figs. 1D-G)
- Uncertainty and novelty based decision bias as a function of EV difference quantile
- Effect of trial horizon on preference
- Effect of variable on choice (RL model-agnostic logistic regression)
- Decision as a function of EV/uncertainty/novelty differences

## compareModelsForSignificantNeurons.m
This function compares how well preselected neurons were explained by different model designs of choice (relevant for Figs 3b,c; 5g-i).
This model comparison approach allows for determining whether neurons were better explained by a full integrated utility model or by a restricted model containing only compents of utility.

## compareNeuronsAcrossVariables_polarCoordinates.m 
This function takes raw Poisson GLM encoding results from forwardGLM.m and utilizes a test of polar coordinates to determine overlap between variables. (relevant for Figs. 2k-m, 3g).

## forwardGLM.m
This is the main analysis script for encoding analyses. Allows for selecting GLM specification, time windows, event alignment, and runs Poisson GLM regression on spike counts.

## forwardGLM_control.m
This function takes a GLM and tests a restricted version of it using the same data, for generating restricted control models.

## getAvgRates_casino.m
Helper for getting average firing rates of neurons

## getFitData.m / getLLE_wsls.m
Specifying RL models which incorporate uncertainty and novelty as components and obtaining loglikelihood of decision time series from model estimates.

## transform_variables.m
(For logistic regression behavioral model fitting) This function takes a time series of choices and exposure to stimuli and infers q-values, uncertainty and novelty, agnostic to any model of uncertainty integration. Instead, we assume that q-values and uncertainty/novelty are extracted from the moments of a beta distribution.

