function [rec, prec, T, F, iou] = DrawPRCurve(SMAP, smapSuffix, GT, gtSuffix, targetIsFg, targetIsHigh, color,line)
% Draw PR Curves for all the image with 'smapSuffix' in folder SMAP
% GT is the folder for ground truth masks
% targetIsFg = true means we draw PR Curves for foreground, and otherwise
% we draw PR Curves for background
% targetIsHigh = true means feature values for our interest region (fg or
% bg) is higher than the remaining regions.
% color specifies the curve color

% Code Author: Wangjiang Zhu
% Email: wangjiang88119@gmail.com
% Date: 3/24/2014

% files = dir(fullfile(SMAP, strcat('*', smapSuffix)));
files = dir(fullfile(GT, strcat('*', gtSuffix)));
num = length(files);
if 0 == num
    error('no saliency map with suffix %s are found in %s', smapSuffix, SMAP);
end

%precision and recall of all images
ALLPRECISION = zeros(num, 256);
ALLRECALL = zeros(num, 256);
ALLF = zeros(num, 256);
ALLIOU = zeros(num, 256);
kk = 0;
for k = 1:num
   tic
    smapName = files(k).name;
%     smapImg = imread(fullfile(SMAP, smapName)); 
    gtImg = imread(fullfile(GT, smapName));
    
%     gtName = strrep(smapName, smapSuffix, gtSuffix);
%     gtImg = imread(fullfile(GT, gtNam
    if exist(fullfile(SMAP, strrep(smapName, gtSuffix,smapSuffix)),'file')&&sum(gtImg(:))
        smapImg = imread(fullfile(SMAP, strrep(smapName, gtSuffix,smapSuffix)));
        kk = kk+1;
    else
        continue;
    end
      
      x = double(smapImg(:,:,1));
%       x = 1./(1+exp(-20*(x/255-0.5)));
%        x = x/255;
%       x(x<0.1)=0;
%       x(x>0.2)=1;
      x = uint8(x/max(x(:)) *255);
      smapImg(:,:,1) =x;
        
%     if exist(fullfile(GT, strrep(smapName, smapSuffix, '.png')),'file')
%             gtImg = imread(fullfile(GT, strrep(smapName, smapSuffix, '.png')));
%         elseif exist(fullfile(GT, strrep(smapName, smapSuffix, '.jpg')),'file')
%             gtImg = imread(fullfile(GT, strrep(smapName, smapSuffix, '.jpg')));
%         elseif exist(fullfile(GT, strrep(smapName, smapSuffix, '.bmp')),'file')
%             gtImg = imread(fullfile(GT, strrep(smapName, smapSuffix, '.bmp')));
%     end
    
    [precision, recall, iou] = CalPR(smapImg, gtImg, targetIsFg, targetIsHigh);
    ALLPRECISION(kk, :) = precision;
    ALLRECALL(kk, :) = recall;
    ALLF(kk, :) = (1.3*precision.*recall)./(0.3*precision+recall);
    ALLIOU(kk, :) = iou;
    t=toc
end
ALLPRECISION = ALLPRECISION(1:kk,:);
ALLRECALL = ALLRECALL(1:kk,:);
ALLF = ALLF(1:kk,:);
ALLIOU = ALLIOU(1:kk,:);
ALLF(isnan(ALLF))=0;
% ALLPRECISION(isnan(ALLPRECISION))=0;
ALLRECALL(isnan(ALLRECALL))=0;
prec = mean(ALLPRECISION, 1);   %function 'mean' will give NaN for columns in which NaN appears.
rec = mean(ALLRECALL, 1); 
f= mean(ALLF, 1);
ALLIOU(isnan(ALLIOU))=0;
iou = mean(ALLIOU,1);
T = [0:255];
F = f;
%plot
% if nargin > 6
%     plot(rec, prec, color,'LineStyle',line ,'linewidth', 2);
% else
%     plot(rec, prec, 'r','LineStyle',line, 'linewidth', 2);
% end
% axis([0,255,0,0.9]);%ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½á·¶Î?
% set(gca, 'XTick', [0 51 102 153 204 255]);