% Plot generated transport operator paths
folderUse = '../results/fmnist/fmnist_M16_z10_zeta0.5_gamma2e-05_test/';
% folderUse = '../results/mnist/mnist_M16_z10_zeta0.1_gamma2e-06_test/';
numDigit = 10;


for m = 1:16
    % Define parameters
    load([folderUse 'transOptOrbitTest_finetune_1.mat']);
    M = size(imgOut,1);
    numStep = size(imgOut,2);
    imgSize = size(imgOut,3);
    c_dim = size(imgOut,5);
    % Specify number of steps to plot
    stepUse = 1:3:numStep;
    imgAll = zeros(numDigit*imgSize,length(stepUse)*imgSize);
    
    
    % Compile generated images
    for n = 1:numDigit
        load([folderUse 'transOptOrbitTest_finetune_' num2str(n) '.mat']);
        
        if n == 3
            title(['Transport Operator ' num2str(m)]);
        end
        count = 1;
        for k = stepUse
            imgAll((n-1)*imgSize+1:n*imgSize,(count-1)*imgSize+1:count*imgSize) = reshape(imgOut(m,k,:,:,:),imgSize,imgSize,c_dim);
            
            count = count+1;
            test = 1;
        end
    end
    
    % Plot generated images
    figure('Position',[30 30 100*length(stepUse) 100*numDigit]);
    imagesc(imgAll)
    axis off
    colormap('gray');
    caxis([0 1])
    title(['Transport Operator ' num2str(m)]);
    fontSize = 20;
    set(gca,'FontSize', fontSize)
    
    saveas(gcf,[folderUse 'transformImg_finetune_TO' num2str(m) '_small.png']);
   
    fprintf('transOpt %d\n', m);
    close all;
end



