% Set folder
folderUse = '../results/fmnist/fmnist_M16_z10_zeta0.5_gamma2e-05_test/';
% folderUse = '../results/mnist/mnist_M16_z10_zeta0.1_gamma2e-06_test/';

% Load test data
imgSize = 28;
load([folderUse 'sampleTest_coeffEncode.mat']);


optUse = 1:16;
% Plot 30 examples
for n = 1:30
    
    % Compile large image of sampled images
    exNum = n;
    imgSampleTotal = zeros(2*imgSize,9*imgSize);
    imgSampleTotal(1:imgSize,1:imgSize) = reshape(x0(exNum,:,:),imgSize,imgSize);
    imgSampleTotal(imgSize+1:2*imgSize,1:imgSize) = reshape(x0(exNum,:,:),imgSize,imgSize);
    for k = 1:8
        imgSampleTotal(1:imgSize,(k)*imgSize+1:(k+1)*imgSize) = reshape(sampled_x(exNum,k,:,:),imgSize,imgSize);
        imgSampleTotal(imgSize+1:2*imgSize,(k)*imgSize+1:(k+1)*imgSize) = reshape(sampled_x_fix(exNum,k,:,:),imgSize,imgSize);
    end
    
    figure('Position',[300,300,1600,600]);imagesc(imgSampleTotal);
    axis off; colormap('gray');
    
    
    saveas(gcf,[folderUse 'sampledOutputs_coeffEncode_' num2str(n) '.fig']);
    saveas(gcf,[folderUse 'sampledOutputs_coeffEncode_' num2str(n) '.png']);
    
    test = 1;
    close all;
end