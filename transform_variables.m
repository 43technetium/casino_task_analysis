% This function takes a time series of choices and exposure to stimuli and
% infers q-values, uncertainty and novelty, agnostic to any model of
% uncertainty integration. Instead, we assume that q-values and
% uncertainty/novelty are extracted from the moments of a beta
% distribution.
function [qVals,uVal,nVal_old] = transform_variables(latents)
    qVals = latents.qVals;
    uVal = latents.uVal;
    nVal = latents.nVal;
    nVal_old = latents.nVal;
    for tI = 1:length(latents.qVals)
        a = (latents.winHistory_block(tI,:)+1);
        b = (latents.selectHistory_block(tI,:)-latents.winHistory_block(tI,:)+1);
        qVals(tI,:) = a./(a+b);
        uVal(tI,:) = (a.*b)./(((a+b).^2).*(a+b+1));
        
        a_n = latents.trialStimHistory_task(tI,:)+1;
        b_n = ones(size(a_n));
        nVal_old(tI,:) = exp(-latents.trialStimHistory_task(tI,:));
        nVal(tI,:) = (a_n.*b_n)./(((a_n+b_n).^2).*(a_n+b_n+1));
    end
end