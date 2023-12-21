function [TPR, FPR, Precision, AUC, AP, F] = ROC( img, hsegmap, NT )
%����ĳһ��sm��ROC������ݣ�������ҪAlgorithm_ROC.m�ṩground truth��
%input parameter description: 
%image�������sm
%hsegmap��sm��Ӧ���ֶ��ָ�ͼ
%NT: �ж��ټ��Ҷ���ֵ
%output parameter description: 
%TPR, FPR�������ʣ������ʣ�1*102
%AUC��ROC�����°����������������ֵ
betaSqr=0.3;

%img=uint8(img*(NT-1));


% hsegmap=double(hsegmap>255/2);
% img=mat2gray(imresize(img,size(hsegmap)));
img=img*(NT-1);

positiveset  = hsegmap; %�ֶ��ָ�ͼ���漯��
negativeset = ~hsegmap ;%�ֶ��ָ�ͼ�ļټ���
P=sum(positiveset(:));%�ֶ��ָ�ͼ���漯�ϵ�ĸ���
N=sum(negativeset(:));%�ֶ��ָ�ͼ�ļټ��ϵ�ĸ���

%��ʼ��TPR��FPR����Ϊ�ж��ٸ���ֵ���о��ж��ٶ�[ TPR, FPR ]
TPR=zeros(1,NT);
FPR=zeros(1,NT);

Precision=zeros(1,NT);
F=zeros(1,NT);
%ȷ����λ��1��0�������Ӱ��÷֣�ֻ����ʾ����ʱ�ÿ�
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
%ͳ�Ƹ���ָ��ľ�����ֵ
      TP=sum(TPmat(:));
      FP=sum(FPmat(:));
%���������ʺͼ�����
      TPR(i)=TP/P;
      FPR(i)=FP/N;
      
      Precision(i)=TP/PS;
      F(i)=(1+betaSqr)*TPR(i)*Precision(i)/(TPR(i)+betaSqr*Precision(i));
       end
end


%����AUC��ROC�����µ������
AUC = -trapz([1,FPR,0], [1,TPR,0]);
AP = -trapz([1,TPR,0], [0,Precision,1]);
end
