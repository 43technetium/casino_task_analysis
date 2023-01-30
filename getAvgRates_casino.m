% This function gets the average firing rate for all units in allUnitCell
%% taquino/aug17
function avgRates = getAvgRates_casino(allUnitCell)
nUnits = length(allUnitCell);
avgRates = nan(nUnits,1);
% Looping over units
for uI = 1:nUnits
    allSpikes = allUnitCell{uI}.unreferencedSpikes;
    timeLength = (max(allSpikes)-min(allSpikes))/1e6;
    avgRates(uI) = length(allSpikes)/timeLength;    
end
end