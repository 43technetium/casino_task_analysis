% This function takes the Poisson GLM encoding results from forwardGLM.m 
% and utilizes a test of polar coordinates to determine overlap between
% variables

function compareNeuronsAcrossVariables_polarCoordinates()
dbstop if error
sessions = {'P41CS_100116','P43CS_110816','P47CS_021817','P48CS_030817'... 
    'P49CS_052317','P51CS_070517', ...
    'P53CS_110817','P54CS_012318','P55CS_031418',...
    'P56CS_042118','P56CS_042518','P60CS_100618','P61CS_022119','P62CS_041919', ...
    'P63CS_082519','P64CS_111719','P65CS_013120', ...
    'P67CS_091720','P67CS_092020','P69CS_102320','P70CS_110620','P71CS_112020'};

basefolder = 'F:\casinoTaskAnalysis\patientData\';


%% Variable setup
% Behavioral model, neural model, spike alignment, spike windowing,
% variable number in the model

% design = 'qLeft_utilUncLeft_qRight_utilUncRight_nLeft_nRight'; 
% design = 'qSel_utilUncSel_qRej_utilUncRej_nSel_nRej'; 
% design = 'utilSel_utilRej_side';
% design = 'utilLeft_utilRight_side';
% design = 'outcome_absRPE'; 
% design = 'outcome_qSel_absRPE'; 
% design = 'exploreFlag_utilUncSel'; 
% design = 'qSel_qRej_qUnseen'; 
design = 'utilSel_utilRej_utilUnseen'; 

% Full model
% variableIdx = [2,4]; dof = 2; % q-value 
% variableIdx = [3,5]; dof = 2; % uncertainty bonus
% variableIdx = [6,7]; dof = 2; % novelty

% Utility decision model
% variableIdx = [2,3]; dof = 2; % utility

% Selected model
% variableIdx = [2]; dof = 1; % q-value 
% variableIdx = [3]; dof = 1; % unc. bonus 
% variableIdx = [6]; dof = 1; % novelty 

% Rejected model
% variableIdx = [4]; dof = 1; % q-value 
% variableIdx = [5]; dof = 1; % unc. bonus 
% variableIdx = [7]; dof = 1; % novelty 

% Utility model
% variableIdx = [2]; dof = 1; % utility selected
% variableIdx = [2,3]; dof = 1; % action utility 
% variableIdx = [4]; dof = 1; % side

% Outcome model
% variableIdx = [2]; dof = 1; % Outcome
% variableIdx = [3]; dof = 1; % qSel
% variableIdx = [4]; dof = 1; % absRPE

% Explore model
% variableIdx = [2]; dof = 1; % Explore flag

% Unseen model
variableIdx = [4]; dof = 1; % Unseen utility


% Alignment and windowing
% alignment = 'trial'; windowing = 'standard_trial';
alignment = 'decision'; windowing = 'standard_decision';
% alignment = 'outcome'; windowing = 'standard_outcome';

vInfo = {'nofGate',design,alignment,windowing,variableIdx(1),'regular'}; 
% Degrees of freedom in unrestricted - restricted model
plottedAreas = [1,4,5];
nAreas = length(plottedAreas);
cStr = num2str(variableIdx-1);
cStr(cStr==' ') = '_';

%% Loading data
allUnits = {};
plottedAreas = [1,4,5];
unitResults = struct();
LRstat = [];
LR_pValue = [];
permutation_t1 = [];  
permutation_t2 = [];
uI = 0;
nPermutations = 500;
pValue_all = [];

for sI = 1:size(sessions,2)    
    session = sessions{sI};
    display(['Session number: ' num2str(sI)])
    sessionFolder = [basefolder session '\'];
    load([sessionFolder 'sessionData.mat']) 
    unitCell = sessionData.neuralData.unitCell;
    % Getting average firing rates of all cells from allUnitCell
    avgRates = getAvgRates_casino(unitCell);
    % Firing rate (in Hz) above which units are considered (0.5 is good)
    rateThreshold = 0.5; 
    unitCell(avgRates<rateThreshold) = [];
    allUnits = [allUnits; unitCell];
    data1 = load(fullfile(basefolder,'forward',vInfo{1},vInfo{2},vInfo{3},vInfo{4},['GLMResults_sessionPermutation_' sessions{sI} '.mat']));
    data_control = load(fullfile(basefolder,'forward',vInfo{1},vInfo{2},vInfo{3},vInfo{4},['GLMResults_linear_control_' sessions{sI} '_' cStr '.mat']));
    nUnits = size(data1.forwardData.mdlCell,1);
    % Get position of each unit in polar coordinates
    
    for i = 1:nUnits        
        display(['Session number: ' num2str(sI) ' / Unit number: ' num2str(i)])
        uI = uI+1;
        for rI = 1:nPermutations+1
            m1 = data1.forwardData.mdlCell{i,rI};
    %         m2 = data2.forwardData.mdlCell{i,2};
            mC = data_control.forwardData.mdlCell{i,rI};
            % Parsing true values
            if rI == 1
                unitResults(uI).b1 = m1.Coefficients{variableIdx(1),1};                
                unitResults(uI).t1 = m1.Coefficients{variableIdx(1),3};
                % Measuring encoding overlap across variables
                if length(variableIdx) == 2
                    unitResults(uI).b2 = m1.Coefficients{variableIdx(2),1};
                    unitResults(uI).t2 = m1.Coefficients{variableIdx(2),3};
                    unitResults(uI).theta = atan2(unitResults(uI).b2,unitResults(uI).b1);
                    unitResults(uI).thetaT = atan2(unitResults(uI).t2,unitResults(uI).t1);
                end
            end
            % If model explains activity for this neuron loglikelihood ratio
            % test
            LL1 = m1.LogLikelihood; LLc = mC.LogLikelihood;
            [~,LR_pValue(uI,rI),LRstat(uI,rI),~] = lratiotest(LL1,LLc,dof);
            permutation_t1(uI,rI) = m1.Coefficients{variableIdx(1),3};
            if length(variableIdx) == 2
                permutation_t2(uI,rI) = m1.Coefficients{variableIdx(2),3};
            end
            
        end
        % Check if unit was significant
        pValue = sum(LRstat(uI,1)<LRstat(uI,2:end))/nPermutations;
        unitResults(uI).pValue = pValue;
        if pValue > 0.05
            unitResults(uI).significant = 0;
            unitResults(uI).class = [0,0];
            unitResults(uI).type = 0;
        else
            unitResults(uI).significant = 1;
            if length(variableIdx) == 2
                [unitResults(uI).class,unitResults(uI).type] = ...
                    getUnitClass(unitResults(uI).theta);
            end
        end
        
        % All p-values for randomized neurons
        for rI = 2:nPermutations+1
            pValue_all(uI,rI) = sum(LRstat(uI,rI)<LRstat(uI,setdiff(2:500,rI)))/(nPermutations-1);
        end
        
    end
end
pValue_all = pValue_all(:,2:end);


%% Getting brain areas
areaVec = zeros(length(allUnits),1);
for uI = 1:length(allUnits)
    brainArea = allUnits{uI,1}.unitInfo;
    areaVec(uI) = brainArea(4);
end
% Merging left/right side for all brain areas
mergedAreas = areaVec;
mergedAreas(mod(mergedAreas,2)==0)=mergedAreas(mod(mergedAreas,2)==0)-1;
% Hippocampus/Amygdala/Anterior cingulate/Supplementary motor areas
hipUnits = find(mergedAreas==1); amyUnits = find(mergedAreas==3);
accUnits = find(mergedAreas==5); smaUnits = find(mergedAreas==7);
ofcUnits = find(mergedAreas==9);
unitsByArea = {ofcUnits; amyUnits; hipUnits; accUnits; smaUnits};
areaNames = {'vmPFC','AMY','HIP','dACC','preSMA','all'};


%% Getting stats
if length(variableIdx) == 2
    nTypes = 4; %(A,B,difference,sum)
    sigAreaTypeRatio = zeros(nAreas,nTypes);
    unitTypes = cell2mat({unitResults.type}).';  
end
areaSigRatio_p = zeros(nAreas,nPermutations);
sigUnits = cell2mat({unitResults.pValue}).' < 0.05;
for aI = 1:nAreas
    aID = plottedAreas(aI);
    if aID == 6
        areaUnits = 1:length(allUnits);
    else
        areaUnits = unitsByArea{aID};
    end    
    areaSigUnits = intersect(find(sigUnits),areaUnits);
    areaSigRatio(aI) = length(areaSigUnits)/length(areaUnits); %#ok<AGROW>
    % Getting type of unit (A,B,difference,sum)
    if length(variableIdx) == 2
        for tI = 1:nTypes            
            sigUnitTypes = find(unitTypes(areaSigUnits)==tI);
            sigAreaTypeRatio(aI,tI) = length(sigUnitTypes)/length(areaSigUnits);
        end
    end
    for rI = 1:nPermutations
        sigUnits_p = pValue_all(:,rI) < 0.05;
        areaSigUnits_p = intersect(find(sigUnits_p),areaUnits);
        areaSigRatio_p(aI,rI) = length(areaSigUnits_p)/length(areaUnits);
    end
end

if length(variableIdx) == 2
    permutationResults.sigAreaTypeRatio = sigAreaTypeRatio;    
    permutationResults.unitTypes = unitTypes;
    permutationResults.permutation_t2 = permutation_t2;
end
permutationResults.areaSigRatio = areaSigRatio;
permutationResults.areaSigRatio_p = areaSigRatio_p;
permutationResults.unitResults = unitResults;
permutationResults.LR_pValue = LR_pValue;
permutationResults.LRstat = LRstat;
permutationResults.permutation_t1 = permutation_t1;

savefolder = fullfile(basefolder,'controlGLM',vInfo{1},vInfo{2},vInfo{3},vInfo{4});    
% Make folder if it doesn't exist
if exist(savefolder,'dir')~=7
    mkdir(savefolder);
end
save([savefolder '/sessionPermutationResults_corrected_' cStr '.mat' ],'permutationResults')

p1 = 1-sum(areaSigRatio(1)>areaSigRatio_p(1,:))/500
p2 = 1-sum(areaSigRatio(2)>areaSigRatio_p(2,:))/500 
p3 = 1-sum(areaSigRatio(3)>areaSigRatio_p(3,:))/500

% figure; hold on; histogram(areaSigRatio_p(1,:),50);
% figure; hold on; histogram(areaSigRatio_p(2,:),50);
% figure; hold on; histogram(areaSigRatio_p(3,:),50);

end

% Getting the encoding angle of a significant unit in polar coordinates, between 2
% variables, and deciding if it encodes A,B,A+B,A-B,-A+B,or -A-B.
function [class,type] = getUnitClass(theta)

if theta < pi/8 && theta >= -pi/8
    class = [1,0];
    type = 1;
elseif theta < 3*pi/8 && theta >= pi/8
    class = [1,1];
    type = 4;
elseif theta < 5*pi/8 && theta >= 3*pi/8
    class = [0,1];
    type = 2;
elseif theta < 7*pi/8 && theta >= 5*pi/8
    class = [-1,1];
    type = 3;
elseif (theta <= pi && theta >= 7*pi/8) || (theta > -pi && theta <= -7*pi/8)
    class = [-1,0];
    type = 1;
elseif theta > -7*pi/8 && theta <= -5*pi/8
    class = [-1,-1];
    type = 4;
elseif theta > -5*pi/8 && theta <= -3*pi/8
    class = [0,-1];
    type = 2;
elseif theta > -3*pi/8 && theta < -pi/8
    class = [1,-1];
    type = 3;
end

end
