function [negLLE, fitData] = getLLE_wsls(params, data, fitOpts)
    % stucture to hold fit resulst
    fitData = struct();
    
    % transform the parameters
    fitData.transParams = transformParams(params, fitOpts);
    
    maxHorizon  = max(data.trialID);
    % softmax beta
    smB         = fitData.transParams(1);  
    % 'learning rate' parameter. Determins decay rate of weighting given to previous observations
    rlP         = fitData.transParams(2);
    % novelty initialization intercept, and terminal value
    nI          = fitData.transParams(3);
    nT          = fitData.transParams(4);
    % novelty utility
    nUtilI      = fitData.transParams(5);
    nUtilT      = fitData.transParams(6);
    % uncertainty utility intercept, terminal, blending
    uI          = fitData.transParams(7);
    uT          = fitData.transParams(8);
    uB          = fitData.transParams(9);
    % familiarity gated uncertaity
    fGateI      = fitData.transParams(10);
    % weight given to response action stickiness
    wActRep     = fitData.transParams(11);
    % weight given to stimulus identity stickiness
    wStimRep    = fitData.transParams(12);
    % propensity to use the left (+1) or right (-1) hand
    hI          = fitData.transParams(13);
    % binary flag: use familirity gate?
    uGate       = fitData.transParams(14);
    
    % Number of stimuli
    maxStimID = max(data.trialStimID(:));
    
    % learned stimulus value (left/right options)
    fitData.qVals   = zeros(length(data.trialID), 2);
    fitData.qVals_all   = zeros(length(data.trialID), maxStimID);
    % distributional expectation quantiles
    fitData.Q1   = zeros(length(data.trialID), 2);
    fitData.Q2   = zeros(length(data.trialID), 2);
    fitData.Q3   = zeros(length(data.trialID), 2);
    fitData.Q4   = zeros(length(data.trialID), 2);
    fitData.Q5   = zeros(length(data.trialID), 2);
    
    % RPE associated with the chosen option
    fitData.RPE         = nan(length(data.trialID), 1);
    % KL divergence of each trial (information gain)
    fitData.KL         = nan(length(data.trialID), 1);
    % probability of each option
    fitData.pOption     = nan(length(data.trialID), 2);
    % probability of the chosen option
    fitData.pChoice     = nan(length(data.trialID), 1);
    % uncertainty and novelty feature values
    fitData.uVal        = nan(length(data.trialID), 2);
    fitData.uVal_all = zeros(length(data.trialID), maxStimID);
    fitData.nVal        = nan(length(data.trialID), 2);
    fitData.nVal_all = zeros(length(data.trialID), maxStimID);
    fitData.isNovel     = nan(length(data.trialID), 2);
    % flag noting if the same action was repeated or not
    fitData.prevResp    = zeros(length(data.trialID), 2);
    % number of times each stimulus has been selelected
    fitData.selectHist  = nan(length(data.trialID), 2);
    fitData.rejectHist  = nan(length(data.trialID), 2);
    
    % trajectory across trials for the novelty bias initiation
    nS          = (nT - nI)/maxHorizon;
    fitData.wN  = nI + nS*(data.trialID - 1);
    % for novelty bonus
    nUtilS          = (nUtilT - nUtilI)/maxHorizon;
    fitData.wUtilN  = nUtilI + nUtilS*(data.trialID - 1);
    % trajectory across trials for the utility of uncertainty
    if isnan(uB)
        % linear slope if slope is not explicitly being fit
        uS          = (uT - uI)/maxHorizon;
        fitData.wU  = uI + uS*(data.trialID - 1);
    else
        % sigmoid transition function
        uS          = 1./(1 + uB * exp(data.trialID-1));
        fitData.wU  = uS*uI + (1-uS) * uT;
    end
    
    % longest set of trials (for use in computing outcome history weighting)    
    wWin = (1-abs(rlP)) .^ ((0:(maxHorizon-1)))';
    if isnan(rlP)
        wWin = zeros(size(wWin));
    end
    
    % track history of wins and losses for each stimulus
    winHist     = zeros( maxHorizon, maxStimID );
    lossHist    = zeros( maxHorizon, maxStimID );
    % track selections/rejections for each stimulus
    selectHist  = zeros( 1, maxStimID );
    rejectHist  = zeros( 1, maxStimID );
    % track novelty bias as stimuli are presented
    exposeHist  = zeros( maxHorizon, maxStimID );
    % track exposure history
    numExp      = zeros( 1, maxStimID );
    % Track utility/qVal/uUtil history
    utilHist = zeros( length(data.trialID), maxStimID );
    qValHist = zeros( length(data.trialID), maxStimID );
    uUtilHist = zeros( length(data.trialID), maxStimID );
    old_alpha_all = ones(1,maxStimID);
    old_beta_all = ones(1,maxStimID);
    % loop through all trials
    for tI = 1 : length(data.trialID)
        % re-initialize expected values at the start of each block
        if data.trialID(tI) == 1   
            % reset outcome and exposure history            
            winHist(:)      = 0;
            lossHist(:)     = 0;
            exposeHist(:)   = 0;
            
        end
        
        % was a response made (i.e. valid trial)
        if ~isnan(data.selectedStimID(tI))
            % weighting history for all previously observed outcomes
            wOutcome = flip(wWin(1:data.trialID(tI)-1));
            % weighting history for all observed stimuli
            wExpose = flip(wWin(1:data.trialID(tI)));
            
            % extract win/loss history from previous trials for each stimulus on offer
            winHistStim = winHist(1:data.trialID(tI)-1, data.trialStimID(tI,:));
            lossHistStim = lossHist(1:data.trialID(tI)-1, data.trialStimID(tI,:));
            winHistStim_all = winHist(1:data.trialID(tI)-1, :);
            lossHistStim_all = lossHist(1:data.trialID(tI)-1, :);
            % graft in the exposure point for first stimulus exposure
            exposeHist( data.trialID(tI), data.trialStimID(tI, numExp(data.trialStimID(tI,:)) == 0) ) = fitData.wN(tI);
            exposeHistStim = exposeHist(1:data.trialID(tI), data.trialStimID(tI,:));
            exposeHistStim_all = exposeHist(1:data.trialID(tI), :);
            
            % compute beta parameters as win/loss history
            alpha   = wOutcome' * winHistStim + 1;
            beta    = wOutcome' * lossHistStim + 1;
            % graft in decayed novelty bias
            wExposeHistStim = wExpose' * exposeHistStim;
            % compute parameters for all stimuli
            alpha_all = wOutcome' * winHistStim_all + 1;
            beta_all = wOutcome' * lossHistStim_all + 1;
            wExposeHistStim_all = wExpose' * exposeHistStim_all;
            
            alpha(wExposeHistStim > 0)  = alpha(wExposeHistStim > 0) + wExposeHistStim(wExposeHistStim > 0);
            beta(wExposeHistStim < 0)   = beta(wExposeHistStim < 0) + abs(wExposeHistStim(wExposeHistStim < 0));
            
            alpha_all(wExposeHistStim_all > 0)  = alpha_all(wExposeHistStim_all > 0) + wExposeHistStim_all(wExposeHistStim_all > 0);
            beta_all(wExposeHistStim_all < 0)   = beta_all(wExposeHistStim_all < 0) + abs(wExposeHistStim_all(wExposeHistStim_all < 0));
            
            
            fitData.qVals(tI,:) = alpha ./ (alpha + beta);
            fitData.qVals_all(tI,:) = alpha_all./(alpha_all + beta_all);
            % Distributional RL quantiles
            quantiles = [0.1667 0.3333 0.5000 0.6667 0.8333];            
            fitData.Q1(tI,:) = betainv(quantiles(1),alpha,beta);
            fitData.Q2(tI,:) = betainv(quantiles(2),alpha,beta);
            fitData.Q3(tI,:) = betainv(quantiles(3),alpha,beta);
            fitData.Q4(tI,:) = betainv(quantiles(4),alpha,beta);
            fitData.Q5(tI,:) = betainv(quantiles(5),alpha,beta);
            
            % compute uncertainty using weighted sampling history (normalized to be max of zero)
            normTerm = (1/12);
            fitData.uVal(tI,:) = (alpha .* beta) ./ ( (alpha+beta).^2 .* (alpha + beta + 1) ) / normTerm;
            fitData.uVal_all(tI,:) = (alpha_all .* beta_all) ./ ( (alpha_all+beta_all).^2 .* (alpha_all + beta_all + 1) ) / normTerm;
            
            % flag indicating if this is the first exposure to a stimulus
            fitData.isNovel(tI,:)   = numExp(data.trialStimID(tI, :)) == 0;
            alpha                   = numExp(data.trialStimID(tI, :)) + 1;
            beta_n                    = 1;
            alpha_all_nov =  numExp + 1;
            beta_all_nov                    = ones(1,length(numExp));
            fitData.nVal(tI,:)      = (alpha .* beta_n) ./ ( (alpha+beta_n).^2 .* (alpha + beta_n + 1) ) / normTerm;
            fitData.nVal_all(tI,:)      = (alpha_all_nov .* beta_all_nov) ./ ( (alpha_all_nov+beta_all_nov).^2 .* (alpha_all_nov + beta_all_nov + 1) ) / normTerm;
            
            % choice history for each stimulus
            fitData.selectHist(tI,:) = selectHist(data.trialStimID(tI, :));
            fitData.rejectHist(tI,:) = rejectHist(data.trialStimID(tI, :));
            
            % compute stimulus RPE
            resp    = data.selectedStimID(tI) == data.trialStimID(tI,:);
            reward  = data.reward(tI);
            fitData.RPE(tI) = reward - fitData.qVals(tI, resp);

            % update exposure counts
            numExp(data.trialStimID(tI,:)) = numExp(data.trialStimID(tI,:)) + 1;
            % update win/loss counts
            winHist(data.trialID(tI), data.selectedStimID(tI))    = reward == 1;
            lossHist(data.trialID(tI), data.selectedStimID(tI))   = reward ~= 1;
            % update select/reject counts
            selectHist(data.selectedStimID(tI)) = selectHist(data.selectedStimID(tI)) + 1;
            rejectHist(data.rejectedStimID(tI)) = selectHist(data.rejectedStimID(tI)) + 1;
        end
        if tI > 1
            % Getting stimuli which changed
            D = data.selectedStimID(tI-1);
            if ~isnan(D)
                fitData.KL(tI-1) = log(beta_function(old_alpha_all(D),old_beta_all(D))/beta_function(alpha_all(D),beta_all(D))) - ...
                    (old_alpha_all(D)-alpha_all(D))*psi(alpha_all(D))-(old_beta_all(D)-beta_all(D))*psi(beta_all(D)) + ...
                    (old_alpha_all(D)-alpha_all(D)+old_beta_all(D)-beta_all(D))*psi(alpha_all(D)+beta_all(D));
                old_alpha_all = alpha_all;
                old_beta_all = beta_all;
            else
                fitData.KL(tI-1) = 0;
            end
            
        end
     
    end % for each trial
    
    % familiarity gate (if in use)
    if uGate == 0
        fitData.fGate = ones(size(fitData.nVal));
        fitData.fGate_all = ones(size(fitData.nVal_all));
    else
        fitData.fGate = 1 - (fGateI.*fitData.nVal);
        fitData.fGate_all = 1 - (fGateI.*fitData.nVal_all);
    end
    
    % utility of stimulus novelty
    fitData.nUtil       = fitData.wUtilN .* fitData.nVal;
    % utility of stimulus uncertainty, as gated by familiarity
    fitData.uUtil       = fitData.fGate .* fitData.wU .* fitData.uVal;
    fitData.uUtil_all       = fitData.fGate_all .* fitData.wU .* fitData.uVal_all;
    % weight applied to sticky action response
    respKeyFlag = zeros( length(data.respKey), 2);
    respKeyHolder = data.respKey;
    respKeyHolder( isnan(data.respKey) ) = 1;
    respKeyFlag( sub2ind(size(respKeyFlag), (1:length(respKeyFlag))', respKeyHolder) ) = 1;
    respKeyFlag( isnan(data.respKey), :) = 0;
    fitData.prevRespFlag = [ [0,0]; respKeyFlag(1:end-1,:)] ;
    % map previouse outtome
    prevOutcomeHolder = data.reward;
    prevOutcomeHolder(prevOutcomeHolder == 0) = -1;
    prevOutcomeHolder = [0; prevOutcomeHolder(1:end-1)];
    fitData.prevRespFlag = [ [0,0]; respKeyFlag(1:end-1,:)] ;
    % interaction between previous action and outcome
    fitData.wsls = fitData.prevRespFlag .* prevOutcomeHolder;
    fitData.uActRep = wActRep * fitData.wsls;
    fitData.uActRep( isnan(fitData.uActRep) ) = 0;
    % weight applied to stimulus sampling history
    alpha               = fitData.selectHist + 1;
    beta                = fitData.rejectHist + 1;
    fitData.sampleVal   = alpha ./ (alpha + beta) - 0.5;
    fitData.uStimRep    = wStimRep .*  fitData.sampleVal;
    % combine utilities across features
    fitData.stimUtil    = fitData.qVals + fitData.nUtil + fitData.uUtil + fitData.uActRep + fitData.uStimRep;
    fitData.stimUtil_all = fitData.qVals_all + fitData.uUtil_all;
    % add in the response hand bias
    fitData.stimUtil(:,1)   = fitData.stimUtil(:,1) + hI;
    % softmax choice probability
    fitData.pOption(:,1)    = 1 ./ (1 + exp(smB .* (fitData.stimUtil(:,2) - fitData.stimUtil(:,1))));
    fitData.pOption(:,2)    = 1-fitData.pOption(:,1);
    
    % probability of selecting the chosen option
    fitData.pChoice = fitData.pOption(:,1);
    fitData.pChoice(data.trialStimID(:,2) == data.selectedStimID) = fitData.pOption(data.trialStimID(:,2) == data.selectedStimID,2);
    
    % determin which trials should be included in the fit
    isValidTrial = ~isnan(data.selectedStimID) & data.isFitTrial;
    % adjust 0 probability trials
    fitData.pChoice(fitData.pChoice < eps | isnan(fitData.pChoice) | isinf(fitData.pChoice)) = eps;
    % compute null model negLLE
    fitData.null_negLLE = sum(isValidTrial) * log(0.5);
    fitData.negLLE = sum(log(fitData.pChoice(isValidTrial)));
    fitData.pseudoR = 1 - (fitData.negLLE/fitData.null_negLLE);
    % compute the negative log-like
    negLLE = fitData.negLLE;
end

function B = beta_function(Z,W)
    B = beta(Z,W);
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
    
    % Familiarity gate fit
    transParams(14) = fitOpts.doFit(14);
    %%%%%%%%%%%%%%%%
    % graft in default values for non-fit parameters
    transParams(~doFit) = defaults(~doFit);
end % function