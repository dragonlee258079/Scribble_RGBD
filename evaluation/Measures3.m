%%%%%%%%%%%%%   F-measuremat
IMGS_DIR = 'F:\chenhao\saliencymap\Final\oursodfinal\';
soddir=dir([IMGS_DIR '*.jpg']);
 for j = 1:length(soddir)
   
    soddirname=soddir(j).name
    soddirnamewhole=strcat(IMGS_DIR,soddirname,'\');
    matpath=soddirnamewhole;
     mats= dir(fullfile(soddirnamewhole));
      disp(soddirnamewhole);
% matpath='F:\ZDW\deepResidual\·½·¨¶þ\SOD2\';
%mats=dir([matpath ]);

gtpath='E:\BSD\BinaryMap2\';
tpr=[];
fpr=[];
auc=[];
ap=[];
f=[];
f_tpr=[];
f_pre=[];
for i=3:length(mats)
    img=mat2gray(imread([matpath mats(i).name]));
    
%     [Precision,TPR, FPR, AUC,AP] = QXL_ROC( img, imread([gtpath mats(i).name(1:end-4) '.tif']), 100 );
%     auc=[auc AUC];
%     ap=[ap AP];
    
    [F TPR Precision] = Fmeasure( img, imread([gtpath mats(i).name(1:end-4) '.tif']) );
    f=[f F];
    f_tpr=[f_tpr TPR];
    f_pre=[f_pre Precision];
end
% mean(auc)
% mean(ap)
% mean(F)
mean(f)
mean(f_tpr)
mean(f_pre)
 end