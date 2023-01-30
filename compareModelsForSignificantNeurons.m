% This function compares how well preselected neurons were explained by
% different model designs of choice
% We run this function for neurons which were previously determined to be
% significant candidates and determine whether they're better explained by a
% full or restricted model.

function compareModelsForSignificantNeurons()
% Fitted behavioral model
modelName = 'nofGate';

%% Windowing/alignment for spikes
% plottedConditions = {'standard_trial','trial'};
% nConditions = size(plottedConditions,1);
% conditionLegend = {'trial onset'};

plottedConditions = {'standard_decision','decision'};
conditionLegend = {'pre-decision'};

plottedAreas = [5]; %1=vmPFC; 4=dACC; 5=preSMA

basefolder = 'F:\casinoTaskAnalysis\patientData\';


%% Designs that will determine the neuron preselection criterion
% First design is the one which will inform selected units

% Track q-value neurons and then utility neurons
% criterionDesigns = {'qLeft_utilUncLeft_qRight_utilUncRight_nLeft_nRight', ...
%     'utilLeft_utilRight_side'};
% criterionVariables = {[2,4],[2,3]}; % q-value and utility

% Track q-value neurons and then utility neurons
criterionDesigns = {'qSel_utilUncSel_qRej_utilUncRej_nSel_nRej', ...
    'utilSel_utilRej_side'};
criterionVariables = {[2],[2]}; % q-value and utility


for dI = 1:length(criterionDesigns)
    designControlledVariables = criterionVariables{dI};
    cStr = num2str(designControlledVariables-1);
    cStr(cStr==' ') = '_';
    data = load(fullfile(basefolder,'controlGLM',modelName,criterionDesigns{dI}, ...
        plottedConditions{2},plottedConditions{1}, ...
        ['sessionPermutationResults_corrected_' cStr '.mat']));
    
    %% Getting data from polar analysis
    totalN = length(data.permutationResults.unitResults);
    unitSignificance = zeros(totalN,1);
    for uI = 1:totalN
        unitSignificance(uI) = data.permutationResults.unitResults(uI).significant;
    end
    if dI == 1
        criterionUnits = find(unitSignificance~=0);
    else
        criterionUnits = union(criterionUnits,find(unitSignificance~=0));
    end
end

%% Determining if these neurons are better explained by full model or restricted model

% fullModel = 'qLeft_utilUncLeft_qRight_utilUncRight_nLeft_nRight';
% controlledVariables = [3,5]; % uncertainty and side

fullModel = 'qSel_utilUncSel_qRej_utilUncRej_nSel_nRej';
controlledVariables = [3]; % uncertainty 

cStr = num2str(controlledVariables-1);
cStr(cStr==' ') = '_';
data = load(fullfile(basefolder,'controlGLM',modelName,fullModel, ...
    plottedConditions{2},plottedConditions{1}, ...
    ['sessionPermutationResults_corrected_' cStr '.mat']));
LR_pValue = data.permutationResults.LR_pValue;
LRstat = data.permutationResults.LRstat;
criterion_LR = LR_pValue(criterionUnits,1);
d = load([basefolder 'unitsByArea.mat' ]);
unitsByArea = d.unitsByArea;
areaNames = {'vmPFC','AMY','HIP','dACC','preSMA'};
colorCell = {[0 0.4470 0.7410],[0.8500 0.3250 0.0980]};
integratedUtilityNeurons = {};
for aI = 1:length(plottedAreas)
    figure; hold on;
    aID = plottedAreas(aI);
    plottedUnits = intersect(unitsByArea{aID},criterionUnits);
    restrictedUnits = intersect(plottedUnits,find(LR_pValue(:,1)>=0.05));
    unrestrictedUnits = intersect(plottedUnits,find(LR_pValue(:,1)<0.05));
    integratedUtilityNeurons = [integratedUtilityNeurons; unrestrictedUnits];
    
    savebase = 'F:\casinoTaskAnalysis\patientData\integratedUtilityUnits_review\';
    savefileName = [savebase 'integratedUtility_' fullModel '_' ...
        plottedConditions{1} '_' plottedConditions{2} '_area_' num2str(aID) '.mat'];
    unrestrictedModelUnits=unrestrictedUnits;
    save(savefileName,'unrestrictedModelUnits');
    
    chiSq_restricted = LRstat(restrictedUnits);
    chiSq_unrestricted = LRstat(unrestrictedUnits);
    % Plotting histogram for LR stats
    title([areaNames{aID} ' (' conditionLegend{1} ')'])
    spreadData = [log(chiSq_restricted);log(chiSq_unrestricted)];
    edges = min(spreadData)-1:1:max(spreadData)+1;
    restrictedLine = histogram(log(chiSq_restricted),edges);
    restrictedLine.EdgeColor = colorCell{1};
    unrestrictedLine = histogram(log(chiSq_unrestricted),edges);
    unrestrictedLine.EdgeColor = colorCell{2};
    ylims = ylim;
    xlims = xlim;
    ylims = [0 1.2.*ylims(2)];
    ylim(ylims)
    ylims = ylim;
    xlabel(['log (LR test statistic)'])        
    ylabel(['Number of neurons'])
    totalUnits = length(restrictedUnits)+length(unrestrictedUnits);
    sensitiveCount = length(unrestrictedUnits);
    pval = 1 - binocdf(sensitiveCount,totalUnits,0.05);
    plotText = ['sig. ratio=' num2str(sensitiveCount) '/'  num2str(totalUnits) ' (p=' num2str(pval,3) ')'];
    text(xlims(1),ylims(2),plotText,'VerticalAlignment','top');
%     hL = legend([restrictedLine,unrestrictedLine],{'null','Integrated utility neurons'});
end

%% How do integrated utility neurons encode q-value and uncertainty?
% fullModel = 'qLeft_utilUncLeft_qRight_utilUncRight_nLeft_nRight';
% testedVariables = {[2,4],[3,5]}; % q-value and uncertainty
% variableTitles = {'q-value','unc. bonus'};

fullModel = 'qSel_utilUncSel_qRej_utilUncRej_nSel_nRej';
testedVariables = {[2],[3]}; % q-value and uncertainty
variableTitles = {'q-value','unc. bonus'};

for aI = 1:length(plottedAreas)
    for vI = 1:length(testedVariables)
        figure; hold on;
        designControlledVariables = testedVariables{vI};
        cStr = num2str(designControlledVariables-1);
        cStr(cStr==' ') = '_';
        data = load(fullfile(basefolder,'controlGLM',modelName,fullModel, ...
            plottedConditions{2},plottedConditions{1}, ...
            ['sessionPermutationResults_corrected_' cStr '.mat']));
        iun_t1 = abs(data.permutationResults.permutation_t1(integratedUtilityNeurons{aI},:));
        if length(testedVariables{vI})==2
            iun_t2 = abs(data.permutationResults.permutation_t2(integratedUtilityNeurons{aI},:));
            preferred_t = [];       
            for pI = 1:size(iun_t1,2)            
                preferred_t(:,pI) = max([iun_t1(:,pI),iun_t2(:,pI)],[],2);            
            end
        else
            preferred_t = iun_t1;
        end
        shuffle_line = histogram(mean(preferred_t(:,2:end),1),30);
        m_true = mean(preferred_t(:,1));
        ylims = ylim();
        true_line = plot(m_true.*ones(10,1),linspace(ylims(1),ylims(2),10),'-r','linewidth',2);
        xlabel('Mean |t-score|')
        ylabel('Frequency')
        title([ variableTitles{vI} ' (' plottedConditions{2} ')'])
        hL = legend([shuffle_line,true_line],{'Shuffled','True value'});
    end
end

end