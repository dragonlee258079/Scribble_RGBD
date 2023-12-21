function Performance = evaluate_SO(gtPath,salPath,ext, modelName, datasetName)
    
    Thresholds = 1:-1/255:0;
    betaSqr=0.3;
    imgFiles=dir([salPath '/*' ext]);
    imgNUM=length(imgFiles);

    resTxt = ['Result_txt/' datasetName '_result-overall.txt'];
    fileID = fopen(resTxt,'a');
    
    [threshold_Fmeasure, threshold_Emeasure] = deal(zeros(imgNUM,length(Thresholds)));
        
    [threshold_Precion, threshold_Recall] = deal(zeros(imgNUM,length(Thresholds)));

    [Smeasure, adpFmeasure, adpEmeasure, MAE] =deal(zeros(1,imgNUM));
    
    
    parfor i = 1:imgNUM  %parfor i = 1:imgNUM  You may also need the parallel strategy. 
%         for i = 1:imgNUM   
            fprintf('Evaluating(%s Dataset,%s Model): %d/%d\n',datasetName, modelName, i,imgNUM);
            name = imgFiles(i).name;
            
            %load gt
            gt = imread([gtPath '/' name]);
            
            if (ndims(gt)>2)
                gt = rgb2gray(gt);
            end
            
            if ~islogical(gt)
                gt = gt(:,:,1) > 128;
            end
            
            %load salency
            sal  = imread([salPath '/' name]);
            
            %check size
            if size(sal, 1) ~= size(gt, 1) || size(sal, 2) ~= size(gt, 2)
                sal = imresize(sal,size(gt));
%                 imwrite(sal,[salPath name]);
                fprintf('Error occurs in the path: %s!!!\n', [salPath name]); %check whether the size of the salmap is equal the gt map.
            end
            
            sal = im2double(sal(:,:,1));
            
            %normalize sal to [0, 1]
            sal = reshape(mapminmax(sal(:)',0,1),size(sal));
            Sscore = StructureMeasure(sal,logical(gt));
            Smeasure(i) = Sscore;
            
            % Using the 2 times of average of sal map as the adaptive threshold.
            threshold =  2* mean(sal(:)) ;
            [~,~,adpFmeasure(i)] = Fmeasure_calu(sal,double(gt),threshold);
            
            
            Bi_sal = zeros(size(sal));
            Bi_sal(sal>threshold)=1;
            adpEmeasure(i) = Enhancedmeasure(Bi_sal,gt);
            
            [threshold_F, threshold_E]  = deal(zeros(1,length(Thresholds)));
            [threshold_Pr, threshold_Rec]  = deal(zeros(1,length(Thresholds)));
            
   
%             for t = 1:length(Thresholds)
%                 threshold = Thresholds(t);
% %                 [threshold_Pr(t), threshold_Rec(t), threshold_F(t)] = Fmeasure_calu(sal,double(gt),size(gt),threshold);
% % %                 [threshold_Pr(t), threshold_Rec(t), threshold_F(t)] = Fmeasure_calu_new(sal,double(gt),size(gt),threshold);
% %                 
%                 Bi_sal = zeros(size(sal));
%                 Bi_sal(sal>threshold)=1;
%                 
% %                 Bi_sal = sal > threshold;
%                 threshold_E(t) = Enhancedmeasure(Bi_sal,gt);
%             end
%             toc
            
%             tic;
            threshold_E = Enhancedmeasure_new(sal,gt, Thresholds);
            
%             toc
            
            [threshold_Pr, threshold_Rec, threshold_F] = Fmeasure_calu_new(sal,double(gt),Thresholds);           
            
            threshold_Fmeasure(i,:) = threshold_F;
            threshold_Emeasure(i,:) = threshold_E;
            threshold_Precion(i,:) = threshold_Pr;
            threshold_Recall(i,:) = threshold_Rec;
            
            MAE(i) = mean2(abs(double(logical(gt)) - sal));
            
   end
    

    %Precision and Recall 
    Performance.column_Pr = mean(threshold_Precion,1);
    Performance.column_Rec = mean(threshold_Recall,1);

    %Mean, Max F-measure score
    Performance.column_F = mean(threshold_Fmeasure,1);
    Performance.meanFm = mean(Performance.column_F);
    Performance.maxFm = max(Performance.column_F);

    %Mean, Max E-measure score
    Performance.column_E = mean(threshold_Emeasure,1);
    Performance.meanEm = mean(Performance.column_E);
    Performance.maxEm = max(Performance.column_E);

    %Adaptive threshold for F-measure and E-measure score
    Performance.adpFm = mean2(adpFmeasure);
    Performance.adpEm = mean2(adpEmeasure);

    %Smeasure score
    Performance.S_measure = mean2(Smeasure);

    %MAE score
    Performance.mae = mean2(MAE);

    %Save the mat file so that you can reload the mat file and plot the PR Curve
%     save([ResPath model],'Smeasure', 'mae', 'column_Pr', 'column_Rec', 'column_F', 'adpFm', 'meanFm', 'maxFm', 'column_E', 'adpEm', 'meanEm', 'maxEm');

    fprintf(fileID, '(Dataset:%s; Model:%s) Smeasure:%.3f; MAE:%.3f; adpEm:%.3f; meanEm:%.3f; maxEm:%.3f; adpFm:%.3f; meanFm:%.3f; maxFm:%.3f.\n',datasetName,modelName,Performance.S_measure, Performance.mae, Performance.adpEm, Performance.meanEm, Performance.maxEm, Performance.adpFm, Performance.meanFm, Performance.maxFm);   
end
   
