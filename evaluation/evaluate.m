function evaluate(model_name)
    %evaluate all methods on all datasets
    %clear,clc
    addpath('/home/lilong/Fight_for_Life/Weakly_RGBD/evaluation/toolbox');
    % addpath('./S-measure-master');

	salMapRootDir='/data1/lilong/Weakly_RGBD/WRGBD_SOTA_Maps'

    allDataset = {'DUTLF-Depth', 'LFSD', 'NJUD', 'NLPR', 'ReDWeb-S', 'RGBD135', 'SIP', 'SSD', 'STERE'};
     
    allModel = {model_name};
    recompute = {''};

    ext='.png';
    perf={'S_measure','maxFm','maxEm','mae'};
    perfWrite=perf;
    NT=256; %�ж��ټ��Ҷ���ֵ

    for datasetIdx=1:length(allDataset)

        datasetName=allDataset{datasetIdx};
        disp(datasetName);

        datasetInfo=getSODatasetInfo(datasetName);
        maskPath=datasetInfo.maskPath;

        results=[];
        for modelIdx=1:length(allModel)
            modelName=allModel{modelIdx};
            disp(modelName);
            flag=1;
            results(modelIdx).modelName=modelName;

            for reIdx=1:length(recompute)
                if strcmp(recompute{reIdx},modelName)
                    flag=1;
                    disp(['recompute: ' modelName])
                    break
                end
            end
            if ~flag
                continue
            end
            salMapDir=[salMapRootDir '/' datasetName '/' modelName];
            if exist(salMapDir,'dir')==7 && length(dir(salMapDir))>10
                Performance=evaluate_SO(maskPath,salMapDir,ext, modelName, datasetName);
                results(modelIdx).Performance=Performance;
            else
                results(modelIdx).Performance=[];
            end

        end
    end
end   
    
 