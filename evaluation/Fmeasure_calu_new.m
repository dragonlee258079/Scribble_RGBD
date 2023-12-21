%%
function [PreFtem, RecallFtem, FmeasureF] = Fmeasure_calu_new(sMap,gtMap, Thresholds)
%threshold =  2* mean(sMap(:)) ;

positiveset  = gtMap;
negativeset = ~gtMap ;
P=sum(positiveset(:));
N=sum(negativeset(:));

NT = length(Thresholds);
[PreFtem, RecallFtem]  = deal(zeros(1,NT));
FmeasureF=zeros(1,NT);


for i=1:NT
    threshold = Thresholds(i);
    
    if ( threshold > 1 )
    threshold = 1;
    end
    
    
    positivesamples = sMap >= threshold;
    TPmat=positiveset.*positivesamples;
%     FPmat=negativeset.*positivesamples;

    PS=sum(positivesamples(:));

    TP=sum(TPmat(:));
 
    if TP == 0
        PreFtem(i) = 0;
        RecallFtem(i) = 0;
        FmeasureF(i) = 0;
    else
        PreFtem(i) = TP/PS;
        RecallFtem(i) = TP/P;
        FmeasureF(i) = ( ( 1.3* PreFtem(i) * RecallFtem(i) ) / ( .3 * PreFtem(i) + RecallFtem(i)));
    end
end
%Fmeasure = [PreFtem, RecallFtem, FmeasureF];

