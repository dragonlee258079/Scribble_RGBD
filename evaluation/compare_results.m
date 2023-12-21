%compare results of HPCANet and newHPCA
clear,clc
addpath('../toolbox');
rootDir='D:\NianLiu\HPCANet_Saliency\results\ECSSD';
dir1=[rootDir '\HPCANet'];
dir2=[rootDir '\newHPCA'];
img1=dir([dir1 '\*.png']);
img2=dir([dir2 '\*.png']);
for i=1:length(img1)
    disp(i)
    sm1=imread([dir1 '\' img1(i).name]);
    %sm1=normalize(sm1);
    sm2=imread([dir2 '\' img2(i).name]);
    MAE=mean(abs(sm1(:)-sm2(:)));
    if MAE
    disp(MAE)
    disp(mean(abs(mat2gray(sm1(:))-mat2gray(sm2(:)))))
    subplot(121),imshow(sm1),subplot(122),imshow(sm2)
    end
end