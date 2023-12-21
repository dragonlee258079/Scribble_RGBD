%%
function [PreFtem, RecallFtem, FmeasureF] = Fmeasure_calu(sMap,gtMap, threshold)
%threshold =  2* mean(sMap(:)) ;
if ( threshold > 1 )
    threshold = 1;
end

positiveset  = gtMap;
negativeset = ~gtMap ;
P=sum(positiveset(:));
N=sum(negativeset(:));


if ( threshold > 1 )
threshold = 1;
end
    
    positivesamples = sMap >= threshold;
    TPmat=positiveset.*positivesamples;
%     FPmat=negativeset.*positivesamples;

    PS=sum(positivesamples(:));

    TP=sum(TPmat(:));
 
    if TP == 0
        PreFtem = 0;
        RecallFtem = 0;
        FmeasureF = 0;
    else
        PreFtem = TP/PS;
        RecallFtem = TP/P;
        FmeasureF = ( ( 1.3* PreFtem * RecallFtem ) / ( .3 * PreFtem + RecallFtem));
    end


%Fmeasure = [PreFtem, RecallFtem, FmeasureF];

