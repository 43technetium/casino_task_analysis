% This function takes in data fit from HBI model comparison and organizes
% it into a usable format
function getFitData()
fitFile = 'F:\casinoTaskAnalysis\patientData\behavior\hbi_results.mat';
fitData = load(fitFile);
dataFiles = dir(fullfile(fullfile('F:\casinoTaskAnalysis\patientData\behavior\sessionBehavior\'), '*.mat'));
% load and collate the data
for fI = 1 : length(dataFiles)
    % load the data and flatten
    load( fullfile(dataFiles(fI).folder, dataFiles(fI).name), 'taskStruct');
    subjectData{fI} = flattenTaskData(taskStruct, fI);
    subjectData{fI}.isFitTrial = ~isnan(subjectData{fI}.respKey);
    clear taskStruct flatBlockData flatModelData flatData;
end

fitResults = {};
negLLE = {};
r = [];
fitOpts.defParamVals    = [0, 0, 0,0, 0,0, 0,0,NaN, 0, 0, 0, 0, 0];

% Taking fits from winning model for each session
for sI = 1:22   
    data = subjectData{sI};
    resp = fitData.cbm.output.responsibility(sI,:);
    [~,winningModel] = max(resp);    
    all_sub_params = fitData.cbm.output.parameters{winningModel};
    params = all_sub_params(sI,:);    
    switch winningModel
        case 1
            fitOpts.doFit = logical([1,1,0,0,0,0,0,0,0,0,0,0,0,0]);
        case 2
            fitOpts.doFit = logical([1,1,0,0,0,0,1,0,0,0,0,0,0,0]);
        case 3
            fitOpts.doFit = logical([1,1,1,0,0,0,0,0,0,0,0,0,0,0]);
        case 4
            fitOpts.doFit = logical([1,1,1,0,0,0,1,0,0,0,0,0,0,0]);
    end
    [negLLE{sI}, fitResults{sI}] = getLLE_wsls(params, data, fitOpts);
    r(sI) = fitResults{1, sI}.pseudoR;
    fitResults{sI}.params = params;
    fitResults{sI}.transfParams = transformParams(params,fitOpts);
end
fitResults = fitResults.';
save('F:\casinoTaskAnalysis\patientData\behavior\allBehavior.mat', 'fitResults')
end


function transParams = transformParams(params,fitOpts)
    defaults = fitOpts.defParamVals;
    doFit = fitOpts.doFit;
    
    % holds the transformed parameters
    transParams = defaults;
    % graft raw parameters into place for those being fit in prep for transformations
    transParams(doFit) = params;
    
    % softmax beta: [0-->inf]
    transParams(1) = exp(transParams(1));
    
    %%%%%%%%%%%%%%%
    % learning rate [0 --> 1]
    transParams(2) = 1./(1+exp(-transParams(2)));
    
    %%%%%%%%%%%%%%%%
    % novelty bias intercept and terminal
    % if not fit, match terminal to novelty intercept (i.e. no slope)
    if ~doFit(4)
        defaults(4) = transParams(3);
    end
    
    %%%%%%%%%%%%%%%%
    % novelty bias intercept and terminal
    % if not fit, match terminal to novelty intercept (i.e. no slope)
    if ~doFit(6)
        defaults(6) = transParams(5);
    end
    
    %%%%%%%%%%%%%%%%
    % uncertainty intercept and terminal
    % if not fit, match terminal to intercept
    if ~doFit(8)
        defaults(8) = transParams(7);
    end
    % start/terminal blending parameter
    transParams(9) = exp(transParams(9));
    
    %%%%%%%%%%%%%%%
    % proportion of uncertainty gated by novelty
    transParams(10) = 1./(1+exp(-transParams(10)));
    
    %%%%%%%%%%%%%%%%
    % graft in default values for non-fit parameters
    transParams(~doFit) = defaults(~doFit);
end % function
