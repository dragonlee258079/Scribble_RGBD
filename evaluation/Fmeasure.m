function [TPR, Precision, F] = Fmeasure( img, hsegmap )
%����ĳһ��sm��ROC������ݣ�������ҪAlgorithm_ROC.m�ṩground truth��
%input parameter description: 
%image�������sm
%hsegmap��sm��Ӧ���ֶ��ָ�ͼ
%NT: �ж��ټ��Ҷ���ֵ


betaSqr=0.3;
%����������sm��fixation map����һ��ͳһ�������ȫ256���Ҷ�ͼ��
%img=mat2gray(image);

% hsegmap=double(hsegmap>255/2);
positiveset  = hsegmap; %�ֶ��ָ�ͼ���漯��
negativeset = ~hsegmap ;%�ֶ��ָ�ͼ�ļټ���
P=sum(positiveset(:));%�ֶ��ָ�ͼ���漯�ϵ�ĸ���
N=sum(negativeset(:));%�ֶ��ָ�ͼ�ļټ��ϵ�ĸ���

%%%%%%%   ����Fmeasure
    %T=min(1,mean(img(:))+std(img(:),0));
    %T=min(1,mean(img(:))+1.2*std(img(:),1));
    Threshold=2*mean(img(:));
%������ֵ�Ĳ��־��������϶����漯��
      positivesamples = img >= Threshold;
%����ground truth��sm��ͼ���  

      TPmat=positiveset.*positivesamples;  
%       TP:true positive ʵ��Ϊ�棬��Ϊ��
      FPmat=negativeset.*positivesamples;
%       FP��false positive �����ԣ��󱨣���ʵ��Ϊ�٣���Ϊ��
       PS=sum(positivesamples(:));
%ͳ�Ƹ���ָ��ľ�����ֵ
      TP=sum(TPmat(:));
      FP=sum(FPmat(:));
%���������ʺͼ�����
      TPR=TP/P;
      FPR=FP/N;
      Precision=TP/PS;
      if PS==0
          F=0;
          Precision=0;
          TPR=0;
      elseif TPR==0
          F=0;
      else
          F=(1+betaSqr)*TPR*Precision/(TPR+betaSqr*Precision);
          %F=TPR*Precision/(0.5*TPR+0.5*Precision);
      end
end




