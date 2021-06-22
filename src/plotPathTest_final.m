folderUse = '../results/fmnist/fmnist_M16_z10_zeta0.5_gamma2e-05_test/';
% folderUse = '../results/mnist/mnist_M16_z10_zeta0.1_gamma2e-06_test/';

imgSize = 28;
channels = 1;
fontSize = 20;
% Specify which classes and corresponding examples top
classUse = 0:9;
exUse = ones(length(classUse),1);


for k = 1:length(classUse)
    % Load data for single class
    load([folderUse 'distTest_singleClass_' num2str(classUse(k)) '.mat']);
    
    z_dim = size(z_trans_out,3);
    numClass = 10;
    numSamp = 10;
    t_temp = 0:0.2:2;
    t_path_len = size(transImgTotal,1);
    interpStep = length(find(t_path <=1));
    
    m = exUse(k);
    imgTotal = zeros(4*imgSize,(t_path_len+1)*imgSize+10,channels);
    % Plot the interpolation and extrapolation sequences
    for n = 1:t_path_len
        if n <=interpStep
            imgTotal(1:imgSize,(n-1)*imgSize+1:n*imgSize,:) = reshape(transImgTotal(n,m,:,:),imgSize,imgSize,channels);
            imgTotal((imgSize+1):2*imgSize,(n-1)*imgSize+1:n*imgSize,:) = reshape(transImgTotal_euc(n,m,:,:),imgSize,imgSize,channels);
            imgTotal(2*(imgSize)+1:3*imgSize,(n-1)*imgSize+1:n*imgSize,:) = reshape(transImgTotal_cae(n,m,:,:),imgSize,imgSize,channels);
            imgTotal(3*(imgSize)+1:end,(n-1)*imgSize+1:n*imgSize,:) = reshape(transImgTotal_bvae(n,m,:,:),imgSize,imgSize,channels);
        else
            imgTotal(1:imgSize,(n)*imgSize+11:(n+1)*imgSize+10,:) = reshape(transImgTotal(n,m,:,:),imgSize,imgSize,channels);
            imgTotal((imgSize+1):2*imgSize,(n)*imgSize+11:(n+1)*imgSize+10,:) = reshape(transImgTotal_euc(n,m,:,:),imgSize,imgSize,channels);
            imgTotal(2*(imgSize)+1:3*imgSize,(n)*imgSize+11:(n+1)*imgSize+10,:) = reshape(transImgTotal_cae(n,m,:,:),imgSize,imgSize,channels);
            imgTotal(3*(imgSize)+1:end,(n)*imgSize+11:(n+1)*imgSize+10,:) = reshape(transImgTotal_bvae(n,m,:,:),imgSize,imgSize,channels);
        end
    end
    
    % Plot the x1 images for comparison
    imgTotal(1:imgSize,(interpStep)*imgSize+1:(interpStep+1)*imgSize,:) = reshape(x1(m,:,:),imgSize,imgSize,channels);
    imgTotal((imgSize+1):2*imgSize,(interpStep)*imgSize+1:(interpStep+1)*imgSize,:) = reshape(x1(m,:,:),imgSize,imgSize,channels);
    imgTotal(2*(imgSize)+1:3*imgSize,(interpStep)*imgSize+1:(interpStep+1)*imgSize,:) = reshape(x1(m,:,:),imgSize,imgSize,channels);
    imgTotal(3*(imgSize)+1:end,(interpStep)*imgSize+1:(interpStep+1)*imgSize,:) = reshape(x1(m,:,:),imgSize,imgSize,channels);
    
    
    imgTotal(1:imgSize,(interpStep+1)*imgSize+1:(interpStep+1)*imgSize+10,:) = ones(imgSize,10);
    imgTotal((imgSize+1):2*imgSize,(interpStep+1)*imgSize+1:(interpStep+1)*imgSize+10,:) = ones(imgSize,10);
    imgTotal(2*(imgSize)+1:3*imgSize,(interpStep+1)*imgSize+1:(interpStep+1)*imgSize+10,:) = ones(imgSize,10);
    imgTotal(3*(imgSize)+1:end,(interpStep+1)*imgSize+1:(interpStep+1)*imgSize+10,:) = ones(imgSize,10);
    
    % Store the probabilities
    prob_temp = reshape(prob_out(:,:),t_path_len,10);
    prob_temp_euc = reshape(prob_out_euc(:,:),t_path_len,10);
    prob_temp_cae = reshape(prob_out_cae(:,:),t_path_len,10);
    prob_temp_bvae = reshape(prob_out_bvae(:,:),t_path_len,10);
    
    figure('Position',[200,200,800,500]);
    subplot(2,1,1);
    imagesc(imgTotal);
    set(gca,'position',[0.13 0.5 0.85 0.45]);
    caxis([0 1])
    colormap('gray');
    axis off;
    subplot(2,1,2);
    plot(t_temp,prob_temp(:,classUse(k)+1),'LineWidth',3);hold all;plot(t_temp,prob_temp_euc(:,classUse(k)+1),'LineWidth',3);plot(t_temp,prob_temp_cae(:,classUse(k)+1),'LineWidth',3);plot(t_temp,prob_temp_bvae(:,classUse(k)+1),'LineWidth',3);ylim([0,1.1]);ylim([0,1.1]);
    set(gca,'position',[0.13 0.18 0.85 0.3]);
    legend('MAE','AE','CAE','\beta-VAE','Location','southwest');
    
    ylabel('Correct Class Prob');
    xlabel('Path Mutliplier');
    xlim([0 t_temp(end)]);
    set(gca,'FontSize', fontSize)
    saveas(gcf,[folderUse 'interpExtrapPath_class' num2str(k) '.png']);
    
    
end


