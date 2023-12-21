clear,clc
smpath1='E:\NianLiu\CNN_for_SO\img_mask\results\BSD\zoomIn\';
smpath2='E:\NianLiu\CNN_for_SO\img_mask\results\BSD\zi_sp\';
gtpath='F:\Public_Data\Saliency\Salient_Object\Benchmarks\BSD\GT\';
sms1=dir([smpath1 '*.png']);
sms2=dir([smpath2 '*.png']);

% Bruce
% load origfixdata
% tpr=[];
% fpr=[];
% auc=[];
% for i=1:length(mats)
%     img=mat2gray(imread([matpath mats(i).name]));
%     fixationPts=white{str2num(mats(i).name(1:end-4))};
%     [Precision,TPR, FPR, AUC] = QXL_ROC( img, fixationPts, 100 );
%     tpr=[tpr;TPR];
%     fpr=[fpr;FPR];
%     auc=[auc AUC];
%     %imwrite(sming,[cd '\result\' gtmats((iter-1)*100+i).name(1:end-4) '.tif']);
% end
% mean(auc)

% tpr=[];
% fpr=[];
% auc=[];
% ap=[];
% pre=[];
apDif=[];
for i=1:length(sms1)
    disp(i/length(sms1));
    img1=mat2gray(imread([smpath1 sms1(i).name]));
    [Precision1,TPR1, FPR1, AUC1,AP1] = QXL_ROC( img1, imread([gtpath sms1(i).name(1:end-4) '.png']), 100 );
    %imwrite(sming,[cd '\result\' gtmats((iter-1)*100+i).name(1:end-4) '.tif']);
    
    img2=mat2gray(imread([smpath2 sms2(i).name]));
    [Precision2,TPR2, FPR2, AUC2,AP2] = QXL_ROC( img2, imread([gtpath sms2(i).name(1:end-4) '.png']), 100 );
    apDif=[apDif;AP2-AP1];
end
[Y,I]=sort(apDif,'ascend');