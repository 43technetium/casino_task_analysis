% This function performs a GLM forward analysis for the casino task

% taquino jan/19

function forwardGLM()
dbstop if error
% % All sessions
sessions = {'P41CS_100116',1;'P43CS_110816',2;'P47CS_021817',3;
    'P48CS_030817',4;'P49CS_052317',5; ...
    'P51CS_070517',6;'P53CS_110817',7;'P54CS_012318',8;'P55CS_031418',9;...
    'P56CS_042118',10;'P56CS_042518',11;'P60CS_100618',12; ...
    'P61CS_022119',13;'P62CS_041919',14; ...
    'P63CS_082519',15; 'P64CS_111719',16; 'P65CS_013120',17; ...
    'P67CS_091720',18; 'P67CS_092020',19; 'P69CS_102320',20;...
    'P70CS_110620',21;'P71CS_112020',22};


behaviorFile = 'allBehavior.mat';
modelName = 'nofGate';


%% Which design matrix to use

% Outcome aligned models
% design = 'outcome';
% design = 'outcome_RPE';
design = 'outcome_qSel_absRPE';
% design = 'outcome_absRPE';

% Model will all decision variables
% design = 'qSel_utilUncSel_qRej_utilUncRej_nSel_nRej';
% design = 'utilLeft_utilRight_side';
% design = 'utilSel_utilRej_side';
% design = 'qLeft_utilUncLeft_qRight_utilUncRight_nLeft_nRight';
% design = 'qUnseen';
% design = 'utilUnseen';

% Models with one side for likelihood ratio comparison
% design = 'qLeft_utilUncLeft_side';
% design = 'qLeft_side';
% design = 'qRight_utilUncRight_side';
% design = 'qRight_side';
% design = 'qLeft_utilUncLeft_nLeft_side';
% design = 'qRight_utilUncRight_nRight_side';


% Control models (Review) full model: 'qLeft_utilUncLeft_qRight_utilUncRight_nLeft_nRight';
% design = 'utilUncLeft_utilUncRight_nLeft_nRight';
% design = 'qLeft_qRight_nLeft_nRight';

% Setting up constant for model
if strcmp(design,'constant') || strcmp(design,'constant_blocks') || strcmp(design,'constant_minUtilTrials')
    interceptFlag = false;
else
    interceptFlag = true;
end

%% Which time alignment to use for spikes
% reference = 'trial';
% reference = 'decision';
reference = 'outcome';

%% Time window arrangement
windowing = 'outcome_windowed'; % Sliding windows for outcome
% windowing = 'decision_windowed'; % Sliding windows for decision
% windowing = 'trial_windowed'; % Sliding windows for trial

% Standard encoding windows
% windowing = 'standard_trial';
% windowing = 'standard_decision';
% windowing = 'standard_post_decision';
% windowing = 'standard_outcome';


%% Loading cells for permutation
basefolder = 'F:\casinoTaskAnalysis\patientData\';

nPermutations = 0;
if nPermutations > 0
    v1 = {modelName,design,reference,windowing,2,'regular'}; 
    d = load(fullfile(basefolder,'neuronCellsForPermutation',v1{1},v1{3},v1{4},'neuronPermutationCell.mat'));  
    permutationData = d.permutationData;
end

for sI = 1:size(sessions,1)    
    session = sessions{sI,1};
    subIdx = sessions{sI,2};
    %% Data setup       
    datafolder = [basefolder session '\behavior\'];
    if isempty(subIdx)
        data = load([datafolder behaviorFile]);
        latents = data.fitResults.latents;
    else
        data = load([basefolder '\allBehavior_intracranial\' behaviorFile]);
        latents = data.fitResults{subIdx};
    end
    sessionFolder = [basefolder session '\'];
    load([sessionFolder 'sessionData.mat']) 
    unitCell = sessionData.neuralData.unitCell;
    sessionRTs = sessionData.trialResponseTime.';
    % Append session number to all units in unitCell
    for unitI = 1:length(unitCell)
        unitCell{unitI}.session = session;
        unitCell{unitI}.sI = sI;
        unitCell{unitI}.unitI = unitI;
    end
    % Getting average firing rates of all cells from allUnitCell
    avgRates = getAvgRates_casino(unitCell);
    % Firing rate (in Hz) above which units are considered (0.5 is good)
    rateThreshold = 0.5; 
    unitCell(avgRates<rateThreshold) = [];
    nTrials = length(unitCell{1,1}.trialReferencedSpikes);
    nUnits = length(unitCell);
    %% Time window setup           
    switch windowing
        case 'standard_decision'
            % Bandit-related time windows
            binSize = 1;
            windowSpacing = 1;
            % For post-reward windows
            period = [-1 0];
            windowStarts = period(1):windowSpacing:period(2)-binSize;
            windowEnds = windowStarts + binSize;
            nBins = length(windowStarts);
        case 'standard_outcome'
            % Bandit-related time windows
            binSize = 1.5;
            windowSpacing = 0.01;
            % For post-reward windows
            period = [0.25 1.75];
            windowStarts = period(1):windowSpacing:period(2)-binSize;
            windowEnds = windowStarts + binSize;
            nBins = length(windowStarts); 
        case 'outcome_windowed'
            % Bandit-related time windows
            binSize = 0.5;
            windowSpacing = 0.016;
            % For post-reward windows
            period = [-0.5 2];
            windowStarts = period(1):windowSpacing:period(2)-binSize;
            windowEnds = windowStarts + binSize;
            nBins = length(windowStarts);
        case 'decision_windowed'
            % Bandit-related time windows
            binSize = 0.5;
            windowSpacing = 0.016;
            % For post-reward windows
            period = [-2 1];
%             binSize = 1;
%             windowSpacing = 0.05;
            % For post-reward windows
%             period = [-2 1];
            windowStarts = period(1):windowSpacing:period(2)-binSize;
            windowEnds = windowStarts + binSize;
            nBins = length(windowStarts);
        case 'trial_windowed'
            % Bandit-related time windows
            binSize = 0.5;
            windowSpacing = 0.016;
            period = [0 2];
%             binSize = 1;
%             windowSpacing = 0.05;
%             period = [-1 2];
            windowStarts = period(1):windowSpacing:period(2)-binSize;
            windowEnds = windowStarts + binSize;
            nBins = length(windowStarts);
        case 'standard_trial'
            % Bandit-related time windows
            binSize = 1.5;
            windowSpacing = 0.02;
            % For post-reward windows
            period = [0.25 1.75001];
            windowStarts = period(1):windowSpacing:period(2)-binSize;
            windowEnds = windowStarts + binSize;
            nBins = length(windowStarts);

    end
    %% Performing forward analysis       
    trialGroups = {};
    % Removing missing trials for P48CS
    fn = fieldnames(latents);
    for k=1:numel(fn)
        field = (latents.(fn{k}));
        if max(size(field))==206
            latents.(fn{k}) = field(1:nTrials,:);
        end
    end
    %% Design matrix setup 
    switch design
        case 'utilSel_utilRej_side'
            side = 2.*latents.selectedVector-3;
            x = [latents.select_stimUtil, latents.reject_stimUtil, side];              
        case 'side'
            side = [2.*latents.selectedVector-3];
            x = side;
        case 'outcome_qSel_absRPE'
            x = [latents.outcome latents.select_qVals abs(latents.RPE)];
        case 'qLeft_utilUncLeft_qRight_utilUncRight_nLeft_nRight'
            x = [ latents.qVals(:,1), latents.uUtil(:,1), ...
                latents.qVals(:,2), latents.uUtil(:,2), ...
                latents.nVal(:,1), latents.nVal(:,2)];
        case 'utilLeft_utilRight_side'
            side = 2.*latents.selectedVector-3;
            x = [latents.stimUtil(:,1), latents.stimUtil(:,2), side];
        case 'qLeft_utilUncLeft_qRight_utilUncRight_side'
            side = 2.*latents.selectedVector-3;
            x = [latents.qVals(:,1) latents.uUtil(:,1) latents.qVals(:,2) latents.uUtil(:,2) side];
        case 'qLeft_qRight_side'
            side = 2.*latents.selectedVector-3;
            x = [latents.qVals(:,1) latents.qVals(:,2) side];
        case 'utilUncLeft_utilUncRight_side'
            side = 2.*latents.selectedVector-3;
            x = [latents.uUtil(:,1), latents.uUtil(:,2), side];
        case 'novLeft_novRight_side'
            side = 2.*latents.selectedVector-3;
            x = [latents.nVal(:,1), latents.nVal(:,2), side];
        case 'exploreFlag_utilUncSel'
            exploreFlag = latents.select_uUtil > latents.reject_uUtil & ...
                latents.select_qVals < latents.reject_qVals;
            x = [exploreFlag, latents.select_uUtil];
        case 'exploitFlag_qSel'
            exploitFlag = latents.select_uUtil < latents.reject_uUtil & ...
                latents.select_qVals > latents.reject_qVals;
            x = [exploitFlag, latents.select_qVals];
        case 'novelFlag_nSel'
            side = latents.selectedVector;
            novelFlag = zeros(length(side),1); 
            for tI = 1:length(side)
                if ~isnan(side(tI))
                    novelFlag(tI) = latents.isNovel(tI,side(tI));
                end
            end
            x = [novelFlag, latents.select_nVal];
        case 'qLeft_utilUncLeft_side'
            side = 2.*latents.selectedVector-3;
            x = [latents.qVals(:,1) latents.uUtil(:,1) side];
        case 'qLeft_side'
            side = 2.*latents.selectedVector-3;
            x = [latents.qVals(:,1) side];
        case 'qRight_utilUncRight_side'
            side = 2.*latents.selectedVector-3;
            x = [latents.qVals(:,2) latents.uUtil(:,2) side];
        case 'qRight_side'
            side = 2.*latents.selectedVector-3;
            x = [latents.qVals(:,2) side];
        case 'qLeft_utilUncLeft_nLeft_side'
            side = 2.*latents.selectedVector-3;
            x = [latents.qVals(:,1) latents.uUtil(:,1) latents.nVal(:,1) side];
         case 'qRight_utilUncRight_nRight_side'
            side = 2.*latents.selectedVector-3;
            x = [latents.qVals(:,2) latents.uUtil(:,2) latents.nVal(:,2) side];
        case 'qUnseen'            
            x = [latents.unseen_qVals];
        case 'utilUnseen'            
            x = [latents.unseen_stimUtil];
        case 'utilUncLeft_utilUncRight_nLeft_nRight'            
            x = [latents.uUtil(:,1), ...
                latents.uUtil(:,2), ...
                latents.nVal(:,1), latents.nVal(:,2)];
        case 'qLeft_qRight_nLeft_nRight'
            x = [latents.qVals(:,1), ...
                latents.qVals(:,2), ...
                latents.nVal(:,1), latents.nVal(:,2)];
            

    end    
    %% Creating model cell (neurons X time bins X (trial groups + all trials))
    nBlocks = max(latents.blockID);
    if strcmp(design, 'utilSel_utilRej_blocks') || strcmp(design, 'utilLeft_utilRight_blocks')
        trialGroups = cell(nBlocks,1);
        blockStims = cell(nBlocks,1);
        nTrialGroups = nBlocks;
        nChosenBlocks = cell(nBlocks,1);
        for blockI = 1:nBlocks
            validTrials = find(~isnan(latents.select_qVals));
            trialGroups{blockI} = intersect(find(latents.blockID==blockI),validTrials);
            blockStims{blockI} = unique([latents.selectStimID(trialGroups{blockI}) ...
                latents.rejectStimID(trialGroups{blockI})]);
            blockStims{blockI} = blockStims{blockI}(~isnan(blockStims{blockI}));
            nBlockStimuli = length(unique(blockStims{blockI}));
            nChosen = zeros(nBlockStimuli,1);
            for stimI = 1:nBlockStimuli
                nChosen(stimI) = sum(latents.selectStimID(trialGroups{blockI})...
                    == blockStims{blockI}(stimI));
            end
            nChosenBlocks{blockI} = nChosen;
        end        
    elseif strcmp(design, 'constant_blocks')
        trialGroups = cell(nBlocks,1);
        nTrialGroups = nBlocks;
        validTrials = find(~isnan(latents.select_qVals));
        for blockI = 1:nBlocks
            trialGroups{blockI} = intersect(find(latents.blockID==blockI),validTrials);
        end
    else
        nTrialGroups = 1;
    end
    mdlCell = cell(nUnits,nBins,nTrialGroups,nPermutations+1);
    bicCell = cell(nUnits,nBins,nTrialGroups,nPermutations+1);
        
    for uI = 1:nUnits         
        display(['Session number: ' num2str(sI) ' / Unit number: ' num2str(uI)])
        % Iterating over time bins and testing GLM
        y = zeros(nTrials,nBins);    
        if strcmp(reference,'outcome')
            spikes = unitCell{uI}.outcomeReferencedSpikes;
        elseif strcmp(reference,'trial')
            spikes = unitCell{uI}.trialReferencedSpikes;    
        elseif strcmp(reference,'decision')
            spikes = unitCell{uI}.decisionReferencedSpikes; 
        end
        % Looping over time bins
        for bI = 1:nBins
            % Dependent variable
            y(:,bI) = cellfun(@(x) nnz(x>windowStarts(bI)&...
                        x<=windowEnds(bI)), spikes);     
            % Models with trial groups
            side = [2.*latents.selectedVector-3];
            final_x = x(~isnan(latents.select_qVals),:); % Remove missed trials
            final_y = y(~isnan(latents.select_qVals),bI); % Remove missed trials
            side = side(~isnan(latents.select_qVals));
%             blockID = blockID(~isnan(latents.select_qVals));
            for gI = 1:nTrialGroups 
                if nTrialGroups > 1
                    % Selecting stimuli that were chosen more than a number of times
                    selectThreshold = 0;
                    if strcmp(design, 'qVals_identity') || strcmp(design, 'stimUtil_identity') || strcmp(design, 'uUtil_identity') 
                        thresholdStimuliIdx = nChosenBlocks{gI} >= selectThreshold;
                        blockStimuli = blockStims{gI};
                        thresholdStimuli = blockStimuli(thresholdStimuliIdx);
                        final_x = x(trialGroups{gI},thresholdStimuli);                                            
                    elseif strcmp(design, 'constant_blocks') || strcmp(design, 'utilSel_utilRej_blocks') || strcmp(design, 'utilLeft_utilRight_blocks')
                        final_x = x(trialGroups{gI});
                    end
                    final_y = y(trialGroups{gI},:);
                end


                if strcmp(design,'utilLeft_utilRight_leftChoice')
                    final_x = final_x(side==-1,:);
                    final_y = final_y(side==-1,:);
                elseif strcmp(design,'utilLeft_utilRight_side_explore')
                    exploreTrials = find(exploreFlag(~isnan(latents.select_qVals)));
                    final_x = final_x(exploreTrials,:);
                    final_y = final_y(exploreTrials,:);
                elseif strcmp(design,'utilLeft_utilRight_side_exploit')
                    exploitTrials = find(exploitFlag(~isnan(latents.select_qVals)));
                    final_x = final_x(exploitTrials,:);
                    final_y = final_y(exploitTrials,:);
                end
                % Model with all trials         
                % Random permutation for boostrapping null distributions                                
                for pI = 1:nPermutations+1                    
                    
                    %% Taking only valid trials
                    criterionTrials = ones(length(final_x),1);                    
                    if subIdx < 12
                        criterionTrials = ~(sessionRTs>Inf|isnan(sessionRTs));
                    end                   
                    permutation_x = final_x(find(criterionTrials),:); %#ok<FNDSB>
                    permutation_y = final_y(find(criterionTrials));                     %#ok<FNDSB>
                    if pI > 1
                        permutation_y = neuronPermutation(permutation_y,sI,permutationData);
                    end
                    
                    %% Running GLM
                    mdl = fitglm(permutation_x,permutation_y,'distribution','Poisson','DispersionFlag',true,'Intercept',interceptFlag);                                
                    mdlCell{uI,bI,gI,pI} = mdl;
                    bicCell{uI,bI,gI,pI} = mdl.ModelCriterion.BIC;
                end
            end
        end
    end
    % Saving forward analysis to file
    forwardData.mdlCell = squeeze(mdlCell);
    forwardData.bicCell = squeeze(bicCell);
    forwardData.reference = reference;
    forwardData.binSize = binSize;
    forwardData.windowSpacing = windowSpacing;
    forwardData.period = period;
    forwardData.windowStarts = windowStarts;
    forwardData.windowEnds = windowEnds;
    forwardData.nBins = nBins; 
    forwardData.session = session; forwardData.subIdx = subIdx; 

    savefolder = [basefolder '/forward/' modelName '/' design '/' reference '/' windowing '/'];
    % Make folder if it doesn't exist
    if exist(savefolder,'dir')~=7
        mkdir(savefolder);
    end
    save([savefolder 'GLMResults_sessionPermutation_' session ],'forwardData')
end
end


% This function obtains data from an eligible random neuron for permutation
function y_p = neuronPermutation(y,sI,permutationData)
    y_p = y;
    % P41-P56
    if permutationData.sessionType(sI) == 1 || permutationData.sessionType(sI) == 2
        eligibleSessionTypes = [1,3];
    % P60-P71
    elseif permutationData.sessionType(sI) == 3
        eligibleSessionTypes = [3];        
    end
    % Eligible neurons by session type
    sT = ismember(permutationData.neuronSessionTypes,eligibleSessionTypes);
    % Eligible neurons by session number
    sN = permutationData.sessionID ~= sI;
    eligibleNeurons = find(sT&sN);
    % Getting random eligible neuron
    chosenNeuron = eligibleNeurons(randi(length(eligibleNeurons)));
    type = permutationData.neuronSessionTypes(chosenNeuron);
    nID = permutationData.nID(chosenNeuron);
    y_new = permutationData.yCell{type}(:,nID);
    if length(y_new) < length(y_p)
        warning('Permutation vector length smaller than original')
    elseif length(y_new) > length(y_p)
        y_p = y_new(1:length(y_p));
    elseif length(y_new) == length(y_p)
        y_p = y_new;
    end
end
