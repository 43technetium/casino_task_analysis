% This function performs behavioral analyses for Aquino et al. (2023), NHB
% Neurons in human pre-supplementary motor area encode key computations for
% value-based choice

function behaviorSummaryPlot()
dbstop if error

%% All sessions and their respective ids
sessions = {'P41CS_100116',1;'P43CS_110816',2;'P47CS_021817',3;
    'P48CS_030817',4;'P49CS_052317',5; ...
    'P51CS_070517',6;'P53CS_110817',7;'P54CS_012318',8;'P55CS_031418',9;...
    'P56CS_042118',10;'P56CS_042518',11;'P60CS_100618',12; ...
    'P61CS_022119',13;'P62CS_041919',14; ...
    'P63CS_082519',15; 'P64CS_111719',16; 'P65CS_013120',17; ...
    'P67CS_091720',18; 'P67CS_092020',19; 'P69CS_102320',20;...
    'P70CS_110620',21;'P71CS_112020',22};

% Behavioral data file
behaviorFile = 'allBehavior.mat';

%% Defining empty vectors for analyses
nQuantiles = 5;
proportionChosen_qVal = zeros(size(sessions,1),nQuantiles);
proportionChosen_qVal_unc = zeros(size(sessions,1),nQuantiles);
proportionChosen_qVal_nov = zeros(size(sessions,1),nQuantiles);
proportionChosen_uVal = zeros(size(sessions,1),nQuantiles);
proportionChosen_nVal = zeros(size(sessions,1),nQuantiles);
proportionChosen_uUtil = zeros(size(sessions,1),nQuantiles);
B = zeros(size(sessions,1),8);
B_r = zeros(size(sessions,1),7);
P = zeros(size(sessions,1),8);
qVal_horizon = zeros(15,size(sessions,1));
unc_horizon = zeros(15,size(sessions,1));
nov_horizon = zeros(15,size(sessions,1));
blockTrialMinMax = zeros(size(sessions,1),2);
percentageMaxUtility = zeros(size(sessions,1),1);
qValUtilCorr = zeros(size(sessions,1),1);
uncBonusUtilCorr = zeros(size(sessions,1),1);
uncBonusRawUncCorr = zeros(size(sessions,1),1);
qValUncCorr = zeros(size(sessions,1),1);
qValNovCorr = zeros(size(sessions,1),1);
uncNovCorr = zeros(size(sessions,1),1);

figure; hold on;
for sI = 1:size(sessions,1)    
    subIdx = sessions{sI,2};
    %% Data setup
    basefolder = 'F:\casinoTaskAnalysis\patientData\';
    data = load([basefolder '\allBehavior_intracranial\' behaviorFile]);
    latents = data.fitResults{subIdx};
    
    % Obtaining model-agnostic q-values, uncertainty and novelty, from past
    % history of rewards, choices and past exposures, respectively
    [qVals,uVal,nVal] = transform_variables(latents);
    
    %% Getting covariate differences for regression
    % Plotting how uncertainty and novelty bias choices
    uUtil_diff = latents.uUtil(:,1) - latents.uUtil(:,2);
    qVal_diff = qVals(:,1) - qVals(:,2);
    uVal_diff = uVal(:,1) - uVal(:,2);
    nVal_diff = nVal(:,1) - nVal(:,2);            
    qVal_diff_unc = zeros(length(qVals),1);
    qVal_diff_nov = zeros(length(qVals),1);    
    isUnc = zeros(length(qVals),1); 
    isNov = zeros(length(qVals),1);     
    for tI = 1:length(qVals)
        [~,uncertainOption] = max(uVal(tI,:));
        [~,novelOption] = max(nVal(tI,:));
        qVal_diff_unc(tI) = qVals(tI,uncertainOption) - qVals(tI,3-uncertainOption);
        qVal_diff_nov(tI) = qVals(tI,novelOption) - qVals(tI,3-novelOption);
        isUnc(tI) = uncertainOption == latents.selectedVector(tI);
        isNov(tI) = novelOption == latents.selectedVector(tI);
    end
    isUnc = logical(isUnc);
    isNov = logical(isNov);
    
              

    %% Performing logistic regression for this session
    % Getting proportion of choices as a function of trial horizon
    trialBlockID = getTrialBlockID(latents.blockID);
    isLeft = latents.selectedVector==1;   
    qVal_trial = qVal_diff.*trialBlockID;
    uVal_trial = uVal_diff.*trialBlockID;
    nVal_trial = nVal_diff.*trialBlockID;  
    zscor_xnan = @(x) bsxfun(@rdivide, bsxfun(@minus, x, mean(x,'omitnan')), std(x, 'omitnan'));
    X = zscor_xnan([qVal_diff uVal_diff nVal_diff trialBlockID qVal_trial uVal_trial nVal_trial]);
    [B(sI,:),~,stats] = mnrfit(X,latents.selectedVector);
    X_r = zscor_xnan([qVal_diff uVal_diff nVal_diff qVal_trial uVal_trial nVal_trial]);
    % Logistic regression
    [B_r(sI,:),~,~] = mnrfit(X_r,latents.selectedVector);
    P(sI,:) = stats.p;
    
    % Plotting how uncertainty and novelty bias choices
    binned_qVal = quantileranks(qVal_diff, nQuantiles, 1);
    binned_uVal = quantileranks(uVal_diff, nQuantiles, 1);
    binned_nVal = quantileranks(nVal_diff, nQuantiles, 1);
    % Getting qVal bins for uncertain and novel options
    binned_qVal_unc = quantileranks(qVal_diff_unc, nQuantiles, 1);
    binned_qVal_nov = quantileranks(qVal_diff_nov, nQuantiles, 1); 
    binned_uUtil = quantileranks(uUtil_diff, nQuantiles, 1);
    for qI = 1:nQuantiles
        proportionChosen_qVal(sI,qI) = sum(binned_qVal(isLeft)==qI)/sum(binned_qVal==qI);
        proportionChosen_uVal(sI,qI) = sum(binned_uVal(isLeft)==qI)/sum(binned_uVal==qI);
        proportionChosen_nVal(sI,qI) = sum(binned_nVal(isLeft)==qI)/sum(binned_nVal==qI);
        proportionChosen_uUtil(sI,qI) = sum(binned_uUtil(isLeft)==qI)/sum(binned_uUtil==qI);
        proportionChosen_qVal_unc(sI,qI) = sum(binned_qVal_unc(isUnc)==qI)/sum(binned_qVal_unc==qI);
        proportionChosen_qVal_nov(sI,qI) = sum(binned_qVal_nov(isNov)==qI)/sum(binned_qVal_nov==qI);
    end
    %% Measuring effect of trial horizon on covariate preference
    [~,~,ic] = unique(latents.blockID);
    a_counts = accumarray(ic,1);
    blockTrialMinMax(sI,1) = min(a_counts);
    blockTrialMinMax(sI,2) = max(a_counts);
    for tI = 1:15
        selectedTrials = trialBlockID == tI;
        selectedTrial_LEFT = isLeft(selectedTrials); 
        selectedTrial_qVals = qVal_diff(selectedTrials); 
        selectedTrial_unc = uVal_diff(selectedTrials);
        selectedTrial_nov = nVal_diff(selectedTrials);
        
        positive_qVals = selectedTrial_qVals>0; choseHigherSide_qVal = selectedTrial_LEFT == positive_qVals;        
        positive_unc = selectedTrial_unc>0; choseHigherSide_unc = selectedTrial_LEFT == positive_unc;        
        positive_nov = selectedTrial_nov>0; choseHigherSide_nov = selectedTrial_LEFT == positive_nov;        
        qVal_horizon(tI,sI) = sum(choseHigherSide_qVal)./length(choseHigherSide_qVal);
        unc_horizon(tI,sI) = sum(choseHigherSide_unc)./length(choseHigherSide_unc);
        nov_horizon(tI,sI) = sum(choseHigherSide_nov)./length(choseHigherSide_nov);
    end
    
    qValUtilCorr(sI) = corr(latents.qVals(:), latents.stimUtil(:),'Rows','complete');
    uncBonusUtilCorr(sI) = corr(latents.uUtil(:), latents.stimUtil(:),'Rows','complete');
    uncBonusRawUncCorr(sI) = corr([latents.uVal(:,1); latents.uVal(:,2)], ...
        [latents.uUtil(:,1); latents.uUtil(:,2)],'Rows','complete');
    qValUncCorr(sI) = corr(latents.qVals(:), latents.uUtil(:),'Rows','complete');
    qValNovCorr(sI) = corr(latents.qVals(:), latents.nVal(:),'Rows','complete');
    uncNovCorr(sI) = corr(latents.uUtil(:), latents.nVal(:),'Rows','complete');
    
    %% Plotting q-val. difference and uncertainty difference for all choices 
    % Plotting "explore choices" in red (selected option has higher
    % uncertainty, lower q-value)
    % Non-explore choices in black
    subplot(5,5,sI); hold on;
    qVal_diff_model = latents.select_qVals - latents.reject_qVals;
    uVal_diff_model = latents.select_uVal - latents.reject_uVal;
    exploreTrials = qVal_diff_model < 0 & uVal_diff_model > 0;
    plot(qVal_diff_model,uVal_diff_model,'k.')
    plot(qVal_diff_model(exploreTrials),uVal_diff_model(exploreTrials),'r.')
    xl = xlim();
    yl = ylim();
    xlabel('q-val. difference')
    ylabel('unc. difference')
    title(['Session ' num2str(sI)])
    plot(linspace(xl(1),xl(2),10),zeros(10,1),'-k')
    plot(zeros(10,1),linspace(yl(1),yl(2),10),'-k')
    xlim(xl)
    ylim(yl)
    
end
%% Utility correlation with q-value/uncertainty bonus
figure; hold on;
subplot(1,2,1); hold on;
histogram(qValUtilCorr,15)
title('utility and q-value')
xlabel('Correlation')
ylabel('Frequency')
qValUtilCorr_m = mean(qValUtilCorr);
qValUtilCorr_s = std(qValUtilCorr)/sqrt(length(qValUtilCorr));

subplot(1,2,2); hold on;
histogram(uncBonusUtilCorr,15)
title('utility and unc. bonus')
xlabel('Correlation')
ylabel('Frequency')
uncBonusUtilCorr_m = mean(uncBonusUtilCorr);
uncBonusUtilCorr_s = std(uncBonusUtilCorr)/sqrt(length(uncBonusUtilCorr));

%% Extra plots for covariate correlations
figure; hold on;
subplot(1,3,1); hold on;
histogram(qValUncCorr,10)
title('q-value and unc. bonus')
xlabel('Correlation')
ylabel('Frequency')
qValUncCorr_m = mean(qValUncCorr);
qValUncCorr_s = std(qValUncCorr)/sqrt(length(qValUncCorr));

subplot(1,3,2); hold on;
histogram(qValNovCorr,10)
title('q-value and novelty')
xlabel('Correlation')
ylabel('Frequency')
qValNovCorr_m = mean(qValNovCorr);
qValNovCorr_s = std(qValNovCorr)/sqrt(length(qValNovCorr));

subplot(1,3,3); hold on;
histogram(uncNovCorr,10)
title('unc. bonus and novelty')
xlabel('Correlation')
ylabel('Frequency')
uncNovCorr_m = mean(uncNovCorr);
uncNovCorr_s = std(uncNovCorr)/sqrt(length(uncNovCorr));
%% Choice proportion figure
mean_qVal = mean(proportionChosen_qVal,1); sem_qVal = std(proportionChosen_qVal,[],1)./sqrt(length(sessions));
mean_qVal_unc = mean(proportionChosen_qVal_unc,1); sem_qVal_unc = std(proportionChosen_qVal_unc,[],1)./sqrt(length(sessions));
mean_qVal_nov = mean(proportionChosen_qVal_nov,1); sem_qVal_nov = std(proportionChosen_qVal_nov,[],1)./sqrt(length(sessions));
mean_uVal = mean(proportionChosen_uVal,1); sem_uVal = std(proportionChosen_uVal,[],1)./sqrt(length(sessions));
mean_nVal = mean(proportionChosen_nVal,1); sem_nVal = std(proportionChosen_nVal,[],1)./sqrt(length(sessions));
mean_uUtil = mean(proportionChosen_uUtil,1); sem_uUtil = std(proportionChosen_uUtil,[],1)./sqrt(length(sessions));

figure; hold on;
e_q = errorbar(1:5,mean_qVal,sem_qVal,'-o','MarkerSize',8,'MarkerEdgeColor','b','MarkerFaceColor','b');
e_u = errorbar(1:5,mean_uVal,sem_uVal,'-o','MarkerSize',8,'MarkerEdgeColor','k','MarkerFaceColor','k');
e_n = errorbar(1:5,mean_nVal,sem_nVal,'-o','MarkerSize',8,'MarkerEdgeColor','m','MarkerFaceColor','m');
plot(linspace(1,5,10),0.5.*ones(10,1),'k--','linewidth',2)


e_q.Color = 'b';
e_u.Color = 'k';
e_n.Color = 'm';

title('Decisions as a function of task variables')
legend({'q-val','unc.','nov.'})
ylabel('Proportion of LEFT choices')
xlabel('LEFT-RIGHT difference quantile')

%% Logistic regression figure
figure; hold on;
variableNames = {'expected value','uncertainty','novelty'};
plot(1.*ones(length(sessions),1)+0.*(rand(length(sessions),1)-0.5),B(:,2),'o','MarkerSize',10,'MarkerEdgeColor','k','MarkerFaceColor',[0.4 0.4 0.4])
plot(2.*ones(length(sessions),1)+0.*(rand(length(sessions),1)-0.5),B(:,3),'o','MarkerSize',10,'MarkerEdgeColor','k','MarkerFaceColor',[0.4 0.4 0.4])
plot(3.*ones(length(sessions),1)+0.*(rand(length(sessions),1)-0.5),B(:,4),'o','MarkerSize',10,'MarkerEdgeColor','k','MarkerFaceColor',[0.4 0.4 0.4])
errors = errorbar(1:3,mean(B(:,2:4),1),std(B(:,2:4),[],1)./sqrt(length(sessions)),'.','MarkerSize',8,'MarkerEdgeColor','red','MarkerFaceColor','red');
errors.Color = 'r';
errors.LineWidth = 2;
plot(linspace(0.5,3.5,10),zeros(10,1),'k--')
xticks([1 2 3])
xticklabels(variableNames)
ylabel('Regression coefficient')


%% Plotting violin plot for logistic regression 
figure; hold on;
violin(B_r(:,2:7),'mc',[],'medc',[]);
plotSpread(B_r(:,2:7),'categoryMarkers',{'.'},'categoryColors',{'k'});
% Plot error bars
errors = errorbar([1,2,3,4,5,6],mean(B_r(:,2:7),1),std(B_r(:,2:7),[],1),'.','MarkerSize',8,'MarkerEdgeColor','red','MarkerFaceColor','red');
errors.Color = 'r';
errors.LineWidth = 3;
% Plot dashed line at 0
plot(linspace(0,7,10),zeros(10,1),'--k')
xticks([1 2 3 4 5 6])
xticklabels({'EV','unc.','nov.','EV:t','unc:t','nov:t'})
ylabel('Estimate')
title('Effect of variable on choice')

% Getting stats
[~,p_EV_L,~,stats_EV_L] = ttest(B_r(:,2)); r_EV = normalityVarianceCheck(B(:,2));
[~,p_unc_L,~,stats_unc_L] = ttest(B_r(:,3)); r_unc = normalityVarianceCheck(B(:,3));
[~,p_nov_L,~,stats_nov_L] = ttest(B_r(:,4)); r_nov = normalityVarianceCheck(B(:,4));
[~,p_EVT_L,~,stats_EVT_L] = ttest(B_r(:,5)); r_EVT = normalityVarianceCheck(B(:,5));
[~,p_uncT_L,~,stats_uncT_L] = ttest(B_r(:,6)); r_uncT = normalityVarianceCheck(B(:,6));
[~,p_novT_L,~,stats_novT_L] = ttest(B_r(:,7)); r_novT = normalityVarianceCheck(B(:,7));


%% Plotting decision proportions as a function of trial horizon
figure; hold on;
qVal_horizon_mean = mean(qVal_horizon,2); qVal_horizon_sem = std(qVal_horizon,[],2)./sqrt(size(sessions,1));
unc_horizon_mean = mean(unc_horizon,2); unc_horizon_sem = std(unc_horizon,[],2)./sqrt(size(sessions,1));
nov_horizon_mean = mean(nov_horizon,2); nov_horizon_sem = std(nov_horizon,[],2)./sqrt(size(sessions,1));
p_q = errorbar(1:length(qVal_horizon_mean),qVal_horizon_mean,qVal_horizon_sem,...
    '-o','MarkerSize',8,'MarkerEdgeColor','b','MarkerFaceColor','b');
p_u = errorbar(1:length(unc_horizon_mean),unc_horizon_mean,unc_horizon_sem,...
    '-o','MarkerSize',8,'MarkerEdgeColor','k','MarkerFaceColor','k');
p_n = errorbar(1:length(nov_horizon_mean),nov_horizon_mean,nov_horizon_sem,...
    '-o','MarkerSize',8,'MarkerEdgeColor','m','MarkerFaceColor','m');
p_q.Color = 'b';
p_u.Color = 'k';
p_n.Color = 'm';
title('Effect of trial horizon over variable preference')
ylabel('Proportion of HIGHER variable chosen')
xlabel('Trial number in block')
plot(linspace(1,length(qVal_horizon_mean),10),0.5.*ones(10,1),'k--','linewidth',2)
legend({'q-val','unc.','nov.'})

% Getting stats
trialN = repmat((1:size(qVal_horizon,1)).',size(qVal_horizon,2),1);
trialN_without_first = repmat((1:size(qVal_horizon(2:end,:),1)).',size(qVal_horizon(2:end,:),2),1);;
flat_qVal = qVal_horizon(:);
flat_qVal_without_first = qVal_horizon(2:end,:);
flat_qVal_without_first = flat_qVal_without_first(:);
flat_unc = unc_horizon(:);
flat_nov = nov_horizon(:);

mdl_qVal = fitlm(trialN_without_first,flat_qVal_without_first); p_qVal = mdl_qVal.Coefficients{2,4};
mdl_unc = fitlm(trialN,flat_unc); p_unc = mdl_unc.Coefficients{2,4};
mdl_nov = fitlm(trialN,flat_nov); p_nov = mdl_nov.Coefficients{2,4};

%% Plotting how uncertainty and novelty bias EV based choices
figure; hold on;
e_q = errorbar(1:5,mean_qVal,sem_qVal,'-o','MarkerSize',8,'MarkerEdgeColor','b','MarkerFaceColor','b');
e_u = errorbar(1:5,mean_qVal_unc,sem_qVal_unc,'-o','MarkerSize',8,'MarkerEdgeColor','k','MarkerFaceColor','k');
e_n = errorbar(1:5,mean_qVal_nov,sem_qVal_nov,'-o','MarkerSize',8,'MarkerEdgeColor','m','MarkerFaceColor','m');
plot(linspace(1,5,10),0.5.*ones(10,1),'k--','linewidth',2)

e_q.Color = 'b';
e_u.Color = 'k';
e_n.Color = 'm';

title('Uncertainty and novelty bias choices')
legend({'left option','uncertain option','novel option'})
ylabel('% chosen')
xlabel('EV difference quantile')

% Getting stats
cQ = proportionChosen_qVal(:);
cU = proportionChosen_qVal_unc(:);
cN = proportionChosen_qVal_nov(:);
y = [cU; cQ; cN];

qG = [ones(size(proportionChosen_qVal,1),1) 2.*ones(size(proportionChosen_qVal,1),1) ...
    3.*ones(size(proportionChosen_qVal,1),1) 4.*ones(size(proportionChosen_qVal,1),1) 5.*ones(size(proportionChosen_qVal,1),1)];
quantileGroup = repmat(qG(:),3,1);
variableGroup =  [-1.*ones(size(cQ,1),1);0.*ones(size(cQ,1),1);1.*ones(size(cQ,1),1)];
mdl_bias = fitlm([quantileGroup,variableGroup],y);
[~,p_bias,~,stats_bias] = ttest2(y(variableGroup == -1),y(variableGroup == 1));




end







