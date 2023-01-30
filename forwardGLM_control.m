% This function takes a GLM and tests a restricted version of it using the
% same data, for generating restricted control models
% Adapted for shuffling only controlled regressor columns

function forwardGLM_control()

sessions = {'P41CS_100116','P43CS_110816','P47CS_021817','P48CS_030817'... 
    'P49CS_052317','P51CS_070517', ...
    'P53CS_110817','P54CS_012318','P55CS_031418',...
    'P56CS_042118','P56CS_042518','P60CS_100618','P61CS_022119','P62CS_041919', ...
    'P63CS_082519','P64CS_111719','P65CS_013120', ...
    'P67CS_091720','P67CS_092020','P69CS_102320','P70CS_110620','P71CS_112020'};

basefolder = 'F:\casinoTaskAnalysis\patientData\';

v1 = {'nofGate','qLeft_utilUncLeft_qRight_utilUncRight_nLeft_nRight','trial','standard_trial',2,'regular'}; 
%v1 = {'nofGate','qLeft_utilUncLeft_qRight_utilUncRight_nLeft_nRight','decision','standard_decision',2,'regular'}; 

% Variables to be controlled in the restricted model (disregard constant)
controlledVariables = [1,3];
%controlledVariables = [5,6];

cStr = num2str(controlledVariables);
cStr(cStr==' ') = '_';

for sI = 1:size(sessions,2)
    display(['Session number: ' num2str(sI)])
    session = sessions{sI};
    sessionFolder = fullfile(basefolder,sessions{sI});
    load([sessionFolder '\sessionData.mat']) 
    unitCell = sessionData.neuralData.unitCell;  
    % Getting average firing rates of all cells from allUnitCell
    avgRates = getAvgRates_casino(unitCell);
    % Firing rate (in Hz) above which units are considered (0.5 is good)
    rateThreshold = 0.5; 
    unitCell(avgRates<rateThreshold) = [];
    data1 = load(fullfile(basefolder,'forward',v1{1},v1{2},v1{3},v1{4},['GLMResults_sessionPermutation_' sessions{sI} '.mat']));

    nUnits = size(data1.forwardData.mdlCell,1);
    nPermutations = 500;
    mdlCell = cell(nUnits,nPermutations+1);
    % Get position of each unit in polar coordinates    
    parfor pI = 1:nPermutations+1        
        display(['Session number: ' num2str(sI) ' / Permutation number: ' num2str(pI)])
        for i = 1:nUnits      
            regressors = table2array(data1.forwardData.mdlCell{i,1}.Variables);
            shuffled_regressors = table2array(data1.forwardData.mdlCell{i,pI}.Variables);
            regressors(:,controlledVariables) = shuffled_regressors(:,controlledVariables);
%             xRange = 1:size(regressors,2)-1;
%             reducedRange = setdiff(xRange,controlledVariables);
%             final_x = table2array(regressors(:,reducedRange));            
            final_x = regressors(:,1:end-1);
            final_y = regressors(:,end);
            mdlCell{i,pI} = fitglm(final_x,final_y,'distribution','Poisson','DispersionFlag',true,'Intercept',true);                                
        end
    end
    
    forwardData.mdlCell = mdlCell;
    savefolder = fullfile(basefolder,'forward',v1{1},v1{2},v1{3},v1{4});    
    % Make folder if it doesn't exist
    if exist(savefolder,'dir')~=7
        mkdir(savefolder);
    end
    save([savefolder '/GLMResults_linear_control_' sessions{sI} '_' cStr '.mat' ],'forwardData')
end