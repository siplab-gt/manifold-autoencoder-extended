% Set folder
folderUse = '../results/fmnist/fmnist_M16_z10_zeta0.5_gamma2e-05_test/';
% folderUse = '../results/mnist/mnist_M16_z10_zeta0.1_gamma2e-06_test/';


% Select which operators to plot. If some have magnitudes of zero, you can
% exclude them
optUse = 1:16;


load([folderUse 'coeffScale_coeffEncode.mat']);

M = size(cspread_store,2);
meanCoeff = zeros(10,M);
stdCoeff = zeros(10,M);

% Plot the coefficient scale for evergy sample in a class
figure;
for k = 1:10
    cscale_temp = cspread_store(label == (k-1),:);
    meanCoeff(k,:) = mean(cscale_temp);
    stdCoeff(k,:) = std(cscale_temp);
    subplot(2,5,k);imagesc(cscale_temp);
    caxis([0 0.05]);
    title(['Class ' num2str(k-1)]);
end

% Plot the compiled image of the coefficient scale per operator per class
figure;
imagesc(1:length(optUse),0:9,meanCoeff(:,optUse));
xlabel('Transport Operator Number');
ylabel('Data Class');
title('Encoded Coefficient Scale - MNIST');
colorbar;
caxis([0.0 0.05]);
saveas(gcf,[folderUse 'cSpreadMean.fig']);
saveas(gcf,[folderUse 'cSpreadMean.png']);


% Create Isomap embedding
D_in = pdist2(z_store(1:2500,:),z_store(1:2500,:));

options.dims = [2 3];
options.display = 0;
NN_opt = 7;
for jj = 1:length(NN_opt)
    NN = NN_opt(jj);
    [Y, R, D] = Isomap(D_in, 'k', NN,options);
    
    % Plot embedding colored by the coefficient scale for each operator
    figure;
    for k = 1:16
        subplot(4,4,k);
        scatter(Y.coords{1}(1,:),Y.coords{1}(2,:),10,cspread_store(Y.index,k),'filled');
        caxis([0.0 0.03]);
        axis off;
        title(['TO ' num2str(k)]);
        
    end
    saveas(gcf,[folderUse 'cSpreadEmbed_NN' num2str(NN) '.fig']);
    saveas(gcf,[folderUse 'cSpreadEmbed_NN' num2str(NN) '.png']);
    
    % Plot embeddign colored by class label
    figure;scatter(Y.coords{1}(1,:),Y.coords{1}(2,:),20,label(Y.index),'filled');axis off;
    saveas(gcf,[folderUse 'classEmbed_NN' num2str(NN) '.fig']);
    saveas(gcf,[folderUse 'classEmbed_NN' num2str(NN) '.png']);
    
    % Plot embedding with images
    figure;hold all;
    for n = 1:500
        imagesc(Y.coords{1}(1,n)*300,Y.coords{1}(2,n)*300,(1-flipud(reshape(imgUse(n,:,:),28,28))),[0 1]);colormap('gray');
    end
end
test = 1;



