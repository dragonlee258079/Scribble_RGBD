function funs = exvel_functions
funs.find_end_row=@find_end_row;
funs.find_dataset_col=@find_dataset_col;
end

function rowIdx = find_end_row(excelFile, sheet)
[num, text, raw] = xlsread(excelFile, sheet);
rowIdx=size(raw, 1);
end

function colIdx = find_dataset_col(dataset, metric)

allDatasets = {'DUT-RGBD', 'LFSD', 'NJU2K', 'NLPR', 'RGBD135', 'SIP', 'ReDWeb_S', 'STERE', 'SSD100'};
allMetrics = {'S_measure','maxFm', 'maxEm', 'mae'};

cols = {'C', 'D', 'E', 'F', 'H', 'I', 'J', 'K', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'W', 'X', 'Y', 'Z', 'AB','AC','AD','AE','AG','AH','AI','AJ','AL','AM','AN','AO', 'AQ', 'AR', 'AS', 'AT'};
[bool,datasetInx]=ismember(dataset,allDatasets);
if ~bool
    error(['Invalid dataset name: ' dataset '!'])
end
[bool,metricInx]=ismember(metric,allMetrics);
if ~bool
    error(['Invalid metric name: ' metric '!'])
end
colIdx = cols{(datasetInx-1)*length(allMetrics)+metricInx};
end