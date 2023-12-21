clear
clc

% calculate IOU
ext= '.png';
maskPath = 'D:\wky\datasets\mask_test';
salMapDir = 'D:\wky\test_data\TCL_test_COCO_dataset_mult';
sms=dir([salMapDir '\*' ext]);
imgNum=length(sms);

 idx=0;
 total_iou = 0;
 for i=1:imgNum
     idx = idx+1;
%       b = sms(i).name;
%         a = sms(i).name;
%         a(end-8:end-4) = [];
%         sms(i).name = a;
%      mask=imread([salMapDir '\' b]);
     mask=imread([salMapDir '\' sms(i).name]);
     
     
%      gt=imread([maskPath '\' a]);
     gt=imread([maskPath '\' sms(i).name]);
     mysize = size(gt);
     if numel(mysize)>2
     gt=rgb2gray(gt);
     end
     
     mask = imresize(mask,size(gt));
     
     Threshold=mean(mask(:));
     mask(mask < Threshold) = 0;
     mask(mask >= Threshold) = 255;
     
     gt(gt<mean(gt(:))) = 0;
     gt(gt>mean(gt(:))) = 255;
    
% mask=imread('***');
% % mask(mask < 30) = 0;
% % mask(mask >= 30) = 255;
% gt=imread('***');
% % gt=255*uint8(gt);

intermask=bitand(mask,gt);
unionmask=bitor(mask,gt); 
intermask=logical(intermask);
unionmask=logical(unionmask);
iou=sum(sum(intermask))/sum(sum(unionmask));
total_iou = total_iou + iou;
% fprint('i=',i);
i
 end
 mean_iou = total_iou/imgNum
