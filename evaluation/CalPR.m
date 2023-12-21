function [precision, recall, IOU] = CalPR(smapImg, gtImg, targetIsFg, targetIsHigh)
% Code Author: Wangjiang Zhu
% Email: wangjiang88119@gmail.com
% Date: 3/24/2014
%%
smapImg = smapImg(:,:,1);
% smapImg = (smapImg-min(smapImg(:)))./(max(smapImg(:))-min(smapImg(:)));
% smapImg = uint8(smapImg*255);
if ~islogical(gtImg)
    gtImg = gtImg(:,:,1) > 128;
end
if any(size(smapImg) ~= size(gtImg))
    error('saliency map and ground truth mask have different size');
     gtImg = imresize(gtImg,size(smapImg));
     %smapImg = imresize(smapImg,[250 320]);
end

if ~targetIsFg
    gtImg = ~gtImg;
end

gtPxlNum = sum(gtImg(:));
if 0 == gtPxlNum
%     error('no foreground region is labeled');
end
%%
targetHist = histc(smapImg(gtImg), 0:255, 1);
nontargetHist = histc(smapImg(~gtImg), 0:255, 1);

U = histc(smapImg(:), 0:255);

if targetIsHigh
    targetHist = flipud(targetHist);
    nontargetHist = flipud(nontargetHist);
    U = flipud(U);
end
%%
targetHist = cumsum( targetHist );
nontargetHist = cumsum( nontargetHist );
U = cumsum( U )+gtPxlNum-targetHist;
%%
precision = targetHist ./(targetHist+ nontargetHist+1e-5);
%%
if any(isnan(precision))
    warning('there exists NAN in precision, this is because of your saliency map do not have a full range specified by cutThreshes\n');
end
recall = targetHist / gtPxlNum;
IOU = targetHist./(U+1e-5);