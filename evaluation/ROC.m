function [TPR, FPR, Precision, AUC, AP, F] = ROC( img, hsegmap, NT )
%计算某一幅sm的ROC相关数据，但是需要Algorithm_ROC.m提供ground truth。
%input parameter description: 
%image：输入的sm
%hsegmap：sm对应的手动分割图
%NT: 有多少级灰度阈值
%output parameter description: 
%TPR, FPR：真真率，假真率，1*102
%AUC：ROC曲线下包含的面积，单个数值
betaSqr=0.3;

%img=uint8(img*(NT-1));


% hsegmap=double(hsegmap>255/2);
% img=mat2gray(imresize(img,size(hsegmap)));
img=img*(NT-1);

positiveset  = hsegmap; %手动分割图的真集合
negativeset = ~hsegmap ;%手动分割图的假集合
P=sum(positiveset(:));%手动分割图的真集合点的个数
N=sum(negativeset(:));%手动分割图的假集合点的个数

%初始化TPR和FPR，因为有多少个阈值所有就有多少对[ TPR, FPR ]
TPR=zeros(1,NT);
FPR=zeros(1,NT);

Precision=zeros(1,NT);
F=zeros(1,NT);
%确保首位是1和0，这个不影响得分，只是显示曲线时好看
% TPR(1)=1;
% FPR(1)=1;
% TPR(NT+2)=0;
% FPR(NT+2)=0;
% Precision(1)=0;
% Precision(NT+2)=1;

for i=1:NT
      T=i-1;
      positivesamples = img >= T;

      TPmat=positiveset.*positivesamples;
      FPmat=negativeset.*positivesamples;
      
       PS=sum(positivesamples(:));
       if PS~=0       
%统计各项指标的具体数值
      TP=sum(TPmat(:));
      FP=sum(FPmat(:));
%计算真真率和假真率
      TPR(i)=TP/P;
      FPR(i)=FP/N;
      
      Precision(i)=TP/PS;
      F(i)=(1+betaSqr)*TPR(i)*Precision(i)/(TPR(i)+betaSqr*Precision(i));
       end
end


%计算AUC（ROC曲线下的面积）
AUC = -trapz([1,FPR,0], [1,TPR,0]);
AP = -trapz([1,TPR,0], [0,Precision,1]);
end
