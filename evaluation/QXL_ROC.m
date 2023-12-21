function [TPR, FPR, Precision, AUC, AP, F] = QXL_ROC( image, hsegmap, NT )
%����ĳһ��sm��ROC������ݣ�������ҪAlgorithm_ROC.m�ṩground truth��
%input parameter description: 
%image�������sm
%hsegmap��sm��Ӧ���ֶ��ָ�ͼ
%NT: �ж��ټ��Ҷ���ֵ
%output parameter description: 
%TPR, FPR�������ʣ������ʣ�1*102
%AUC��ROC�����°����������������ֵ
betaSqr=0.3;
%����������sm��fixation map����һ��ͳһ�������ȫ256���Ҷ�ͼ��
img=mat2gray(image);

%img=uint8(img*(NT-1));


hsegmap=double(im2bw(mat2gray(hsegmap),0.5));%���򻯵�[0 1]
hsegmap=hsegmap(:,:,1);
img=imresize(img,size(hsegmap));
img=(img*(NT-1));

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
%�������sm�����ǰ������ȫ100���ҶȺ󣬾���0~99��100���Ҷ���Ϊ��ֵ��������ֵ��ѡȡ���������Թ��ܶ��ַ��������������PAMI2011
%�Ĵ��룬ͳ��sm���м������ظ��ĻҶȣ�����Щ�Ҷ�����ֵ��������������ȷ�ֽ�ʡ�����������ǣ�����ÿ��sm�ﲻ�ظ��ĻҶȸ���������ͬ����
%�������TPR��FPR�ĸ���Ҳ�Ͳ�ͬ������û���Ѹ���sm��TPR��FPR����ֵ�ͷ������㡣����һ�ַ�������sm�������صĻҶȽ������򣬽���
%Щ���ص�ȷֳ�N�ݣ�ÿ��һ����ֵ���ؼ���Nȡ���N���С��sm�Ĳ��ظ��ĻҶȸ�������ôROC��ֵ�����߾ͻ�ƫ�ͣ����������sm�Ĳ���
%���ĻҶȸ�����ÿ���ֲ�ͬ��ֻ�ܰ�Nȡ��һ�㣬��ʤ����ȡN=320�����������˷ѣ���ΪN>256�Ļ���û�����ˡ��Ҿ���ʵ�鷢��ȡ100����ֵ��
%���������������Ա����൱�ľ���
      T=i-1;
%������ֵ�Ĳ��־��������϶����漯��
      positivesamples = img >= T;
%��������ͼ���

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
