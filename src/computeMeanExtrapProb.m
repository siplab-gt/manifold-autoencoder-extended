folderUse = '../results/fmnist/fmnist_M16_z10_zeta0.5_gamma2e-05_test/';
% folderUse = '../results/mnist/mnist_M16_z10_zeta0.1_gamma2e-06_test/';

prob_total = zeros(11,4000);
prob_total_euc = zeros(11,4000);
prob_total_cae = zeros(11,4000);
prob_total_bvae = zeros(11,4000);
acc_out_total = zeros(11,4000);
acc_out_total_euc = zeros(11,4000);
acc_out_total_cae = zeros(11,4000);
acc_out_total_bvae = zeros(11,4000);
for ii = 0:9
    class_start = ii;
    load([folderUse 'extrapProbTest_singleClass_' num2str(class_start) '.mat']);
    
    % Compute the mean probability of the correct label for every class
    mean_prob = mean(prob_out,2);
    mean_prob_euc = mean(prob_out_euc,2);
    mean_prob_cae = mean(prob_out_cae,2);
    mean_prob_bvae = mean(prob_out_bvae,2);
    std_prob = std(prob_out');
    std_prob_euc =std(prob_out_euc');
    std_prob_cae =std(prob_out_cae');
    std_prob_bvae =std(prob_out_bvae');
    
    % Store the probability and the accuracy
    prob_total(:,ii*400+1:(ii+1)*400) = prob_out;
    prob_total_euc(:,ii*400+1:(ii+1)*400) = prob_out_euc;
    prob_total_cae(:,ii*400+1:(ii+1)*400) = prob_out_cae;
    prob_total_bvae(:,ii*400+1:(ii+1)*400) = prob_out_bvae;
    acc_out_total(:,ii*400+1:(ii+1)*400) = acc_out;
    acc_out_total_euc(:,ii*400+1:(ii+1)*400) = acc_out_euc;
    acc_out_total_cae(:,ii*400+1:(ii+1)*400) = acc_out_cae;
    acc_out_total_bvae(:,ii*400+1:(ii+1)*400) = acc_out_bvae;
    test = 1;
end


% Compute the overall mean and standard deviation of the probability of the
% correct label
mean_prob_total = mean(prob_total,2);
mean_prob_euc_total = mean(prob_total_euc,2);
mean_prob_cae_total = mean(prob_total_cae,2);
mean_prob_bvae_total = mean(prob_total_bvae,2);
std_prob_total = std(prob_total');
std_prob_euc_total =std(prob_total_euc');
std_prob_cae_total =std(prob_total_cae');
std_prob_bvae_total =std(prob_total_bvae');

% Compute the mean accuracy to plot
mean_acc = sum(acc_out_total,2)/4000;
mean_acc_euc = sum(acc_out_total_euc,2)/4000;
mean_acc_cae = sum(acc_out_total_cae,2)/4000;
mean_acc_bvae = sum(acc_out_total_bvae,2)/4000;

% Plot the mean accuracy
fontSize = 20;
figure;plot(t_path,mean_acc,'LineWidth',4);hold all;plot(t_path,mean_acc_euc,'LineWidth',3);plot(t_path,mean_acc_cae,'LineWidth',3);plot(t_path,mean_acc_bvae,'LineWidth',3);
ylim([0.3 1]);
legend('MAE','AE','CAE','\beta -VAE','Location','southwest');
xlabel('Path Multiplier');
ylabel('Classification Accuracy');
title('Fashion MNIST');
set(gca,'FontSize', fontSize)
saveas(gcf,[folderUse 'extrapClassAccAll_fmnist.png']);
saveas(gcf,[folderUse 'extrapClassAccAll_fmnist.fig']);