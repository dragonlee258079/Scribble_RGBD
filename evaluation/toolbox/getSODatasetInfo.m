%get dataset info
function datasetInfo=getSODatasetInfo(datasetName)
   
    rootPath = 'path for test data...';
    switch datasetName
		case 'DUT-RGBD'
            rootPath=[rootPath 'DUTLF-Depth'];
            imgPath=[rootPath '/RGB'];
            maskPath=[rootPath '/GT'];
            depthPath=[rootPath '/depth'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
		case 'LFSD'
            rootPath=[rootPath 'LFSD'];
            imgPath=[rootPath '/RGB'];
            maskPath=[rootPath '/ground_truth'];
            depthPath=[rootPath '/depth'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
        case 'NJU2K'
            rootPath=[rootPath 'NJU2D'];
            imgPath=[rootPath '/RGB'];
            maskPath=[rootPath '/GT'];
            depthPath=[rootPath '/depth'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
        case 'NLPR'
            rootPath=[rootPath 'NLPR'];
            imgPath=[rootPath '/RGB'];
            maskPath=[rootPath '/GT'];
            depthPath=[rootPath '/depth'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
		case 'RGBD135'
            rootPath=[rootPath 'RGBD135'];
            imgPath=[rootPath '/RGB'];
            maskPath=[rootPath '/GT'];
            depthPath=[rootPath '/depth'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
		case 'SIP'
            rootPath=[rootPath '/SIP'];
            imgPath=[rootPath '/RGB'];
            maskPath=[rootPath '/GT'];
            depthPath=[rootPath '/depth'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
		case 'SSD100'
            rootPath=[rootPath 'SSD'];
            imgPath=[rootPath '/RGB'];
            maskPath=[rootPath '/GT'];
            depthPath=[rootPath '/depth'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
		case 'STERE'
            rootPath=[rootPath 'STERE'];
            imgPath=[rootPath '/RGB'];
            maskPath=[rootPath '/GT'];
            depthPath=[rootPath '/depth'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
		case 'ReDWeb-S'
            rootPath=[rootPath 'ReDWeb-S'];
            imgPath=[rootPath '/RGB'];
            maskPath=[rootPath '/GT'];
            depthPath=[rootPath '/depth'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
        otherwise
            disp('Invalid dataset name!');
            return
    end

    datasetInfo.rootPath=rootPath;
    datasetInfo.imgPath=imgPath;
    datasetInfo.maskPath=maskPath;
    datasetInfo.depthPath=depthPath;
    datasetInfo.imgFiles=imgFiles;
    datasetInfo.maskFiles=maskFiles;
    datasetInfo.imgNum=length(imgFiles);
end