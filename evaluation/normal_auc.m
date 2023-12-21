clear,clc

%allDataset={'BSD','ECSSD','MSRA10K-test','PASCAL1500','SED'};
allDataset={'BSD','ECSSD','MSRA10K-test','PASCAL1500','PASCAL-S','SED'};
type='RCL1sm';

betaSqr=0.3;

for datasetIdx=1:length(allDataset)
    close all
    
    datasetName=allDataset{datasetIdx};
    disp(datasetName);
    
    datasetInfo=getSODatasetInfo(datasetName);
    gtpath=datasetInfo.maskPath;
    resultsPath=[datasetInfo.resultsPath '\' type '\'];
    sms=dir([resultsPath '*.png']);
    imgNum=length(sms);
    
    tpr=[];
    fpr=[];
    auc=[];
    ap=[];
    pre=[];
    fm=[];
    tpr1=[];
    pre1=[];
%     wP=[];
%     wR=[];
%     wF=[];
    
    for i=1:imgNum
        disp(i/imgNum)
        img=mat2gray(imread([resultsPath sms(i).name]));
        gt=imread([gtpath '\' sms(i).name(1:end-4) '.png']);
        [Prec,TPR, FPR, AUC,AP] = QXL_ROC( img, gt, 100 );
        pre=[pre;Prec];
        tpr=[tpr;TPR];
        fpr=[fpr;FPR];
        auc=[auc;AUC];
        ap=[ap;AP];
        [fmTmp TPR1 Precision1] = Fmeasure( img, gt );
        fm=[fm;fmTmp];
        tpr1=[tpr1;TPR1];
        pre1=[pre1;Precision1];
        
%         [wFTmp,wPTmp,wRTmp]= WFb(img,logical(gt));
%         wF=[wF;wFTmp];
%         wP=[wP;wPTmp];
%         wR=[wR;wRTmp];
        %imwrite(sming,[cd '\result\' gtmats((iter-1)*100+i).name(1:end-4) '.tif']);
    end
%     mean(auc)
%     mean(ap)
%     mean(fm)
    T=mean(tpr,1);
    F=mean(fpr,1);
    P=mean(pre,1);

    AUC = -trapz(F, T)
    AP = -trapz(T, P)

    Precision=mean(pre1)
    Recall=mean(tpr1)
    Fm=mean(fm)
    
%     wPrecision=mean(wP)
%     wRecall=mean(wR)
%     wFm=mean(wF)

%     h=figure(1);
%     plot(T(2:end-1),P(2:end-1));
%     saveas(h,[datasetName '_AP_' type],'fig');
%     
%     h=figure(2);
%     plot(F(2:end-1),T(2:end-1));
%     saveas(h,[datasetName '_AUC_' type],'fig');
end