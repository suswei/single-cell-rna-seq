cd './code';
k = 3;
repos = 100;
samples = 10000;
N = 4; %the dimension of gaussian distribution
   
y0_array = {[ .4 .5 .8 ],[ .4 .5 .8 ], [ .4 .5 .8 ],[ .4 .5 .8 ],[ .4 .5 .8 ],[ .4 .5 .8 ],[ .4 .5 .8 ],[ .3 .5 .9 ],[ 3 5 9 ],[ 3 5 9 ],[ 3 9 18 ],[ 3 9 18 ],[ 5 15 30 ]};          % the center of the gaussian
sigma_y_array = {[ .2 .3 .25 ],[ .2 .3 .25 ], [ .2 .3 .25 ],[ .2 .3 .25 ],[ .2 .3 .25 ],[ .2 .3 .25 ],[ .2 .3 .25 ],[ .2 .3 .25 ],[ .2 .3 .25 ],[ 2 3 2.5 ],[ .2 .3 .25 ],[ .2 .3 .25 ],[ .2 .3 .25 ]};    % the gaussian decay constant
p_array = {[0.1176 0.5882 0.2942],[0.2176 0.4882 0.2942],[0.3176 0.3882 0.2942],[0.4176 0.2882 0.2942],[0.5176 0.1882 0.2942],[0.6176 0.0882 0.2942],[0.7176 0.0882 0.1942],[0.1 0.3 0.6],[0.1 0.3 0.6],[0.1 0.3 0.6],[0.33 0.33 0.34],[0.9 0.05 0.05],[0.33 0.33 0.34]}; 

iterations = length(y0_array);

true_MI = zeros(1,iterations);
estimated_MI = zeros(repos, iterations);

true_MI_dimN = zeros(1,iterations);
estimated_MI_dimN = zeros(repos, iterations);

syms x1 x2 x3 x4

V_dim10 = 0;

for iteration = 1:iterations
    y0 = y0_array{iteration};
    sigma_y = sigma_y_array{iteration};
    p = p_array{iteration};
    cumulative_p = cumsum(p);
    Ay = p./(sqrt(2*pi)*sigma_y);     % MI of one dimension Gaussians
    mu_y = @(y) sum( Ay .* exp(-(y-y0).^2./(2*sigma_y.^2)), 2 );
    Hy = @(y) -mu_y(y) .* log(mu_y(y));

    true_MI(1,iteration) = integral(Hy, min(y0 - 10*sigma_y), max(y0 + 10*sigma_y), ...
        'ArrayValued', true) - 0.5 - sum(p.*log(sqrt(2*pi)*sigma_y));

    true_MI(1,iteration) = true_MI(1,iteration) / log(2);
    
    V = zeros(1, samples);
        
    Cov_X1 = (sigma_y_array{iteration}(1))^2*eye(N);
    Deter_Cov_X1 = det(Cov_X1);    
        
    Cov_X2 = (sigma_y_array{iteration}(2))^2*eye(N);
    Deter_Cov_X2 = det(Cov_X2);    
    
    Cov_X3 = (sigma_y_array{iteration}(3))^2*eye(N);
    Deter_Cov_X3 = det(Cov_X3);
    
    x1_min = min(y0_array{iteration} - 10*sigma_y_array{iteration});
    x1_max = max(y0_array{iteration} + 10*sigma_y_array{iteration});
    
    Hy_gaussian_mixture = @(x1,x2,x3,x4)-(p(1)* ((1/(sqrt(Deter_Cov_X1)*(2*pi)^(N/2)))*exp(-(transpose([x1-y0(1); x2-y0(1); x3-y0(1); x4-y0(1)])/Cov_X1*[x1-y0(1); x2-y0(1); x3-y0(1); x4-y0(1)])/2))+ p(2)* ((1/(sqrt(Deter_Cov_X2)*(2*pi)^(N/2)))*exp(-(transpose([x1-y0(2); x2-y0(2); x3-y0(2); x4-y0(2)])/Cov_X2*[x1-y0(2); x2-y0(2); x3-y0(2); x4-y0(2)])/2)) + p(3)* ((1/(sqrt(Deter_Cov_X3)*(2*pi)^(N/2)))*exp(-(transpose([x1-y0(3); x2-y0(3); x3-y0(3); x4-y0(3)])/Cov_X3*[x1-y0(3); x2-y0(3); x3-y0(3); x4-y0(3)])/2)))*log((p(1)* ((1/(sqrt(Deter_Cov_X1)*(2*pi)^(N/2)))*exp(-(transpose([x1-y0(1); x2-y0(1); x3-y0(1); x4-y0(1)])/Cov_X1*[x1-y0(1); x2-y0(1); x3-y0(1); x4-y0(1)])/2))+ p(2)* ((1/(sqrt(Deter_Cov_X2)*(2*pi)^(N/2)))*exp(-(transpose([x1-y0(2); x2-y0(2); x3-y0(2); x4-y0(2)])/Cov_X2*[x1-y0(2); x2-y0(2); x3-y0(2); x4-y0(2)])/2)) + p(3)* ((1/(sqrt(Deter_Cov_X3)*(2*pi)^(N/2)))*exp(-(transpose([x1-y0(3); x2-y0(3); x3-y0(3); x4-y0(3)])/Cov_X3*[x1-y0(3); x2-y0(3); x3-y0(3); x4-y0(3)])/2))));
    Hy = integralN(@(x1,x2,x3,x4)arrayfun(Hy_gaussian_mixture,x1,x2,x3,x4),x1_min,x1_max,x1_min,x1_max,x1_min,x1_max, x1_min,x1_max);
       
    %Hy_gaussian_mixture = @(x1,x2,x3,x4,x5,x6)-(p(1)* ((1/(sqrt(Deter_Cov_X1)*(2*pi)^(N/2)))*exp(-(transpose([x1-y0(1); x2-y0(1); x3-y0(1);x4-y0(1);x5-y0(1);x6-y0(1)])/Cov_X1*[x1-y0(1); x2-y0(1); x3-y0(1);x4-y0(1);x5-y0(1);x6-y0(1)])/2))+ p(2)* ((1/(sqrt(Deter_Cov_X2)*(2*pi)^(N/2)))*exp(-(transpose([x1-y0(2); x2-y0(2); x3-y0(2); x4-y0(2); x5-y0(2); x6-y0(2)])/Cov_X2*[x1-y0(2); x2-y0(2); x3-y0(2); x4-y0(2); x5-y0(2); x6-y0(2)])/2)) + p(3)* ((1/(sqrt(Deter_Cov_X3)*(2*pi)^(N/2)))*exp(-(transpose([x1-y0(3); x2-y0(3); x3-y0(3); x4-y0(3); x5-y0(3); x6-y0(3)])/Cov_X3*[x1-y0(3); x2-y0(3); x3-y0(3); x4-y0(3); x5-y0(3); x6-y0(3)])/2)))*log(p(1)* ((1/(sqrt(Deter_Cov_X1)*(2*pi)^(N/2)))*exp(-(transpose([x1-y0(1); x2-y0(1); x3-y0(1); x4-y0(1); x5-y0(1); x6-y0(1)])/Cov_X1*[x1-y0(1); x2-y0(1); x3-y0(1); x4-y0(1); x5-y0(1); x6-y0(1)])/2))+ p(2)* ((1/(sqrt(Deter_Cov_X2)*(2*pi)^(N/2)))*exp(-(transpose([x1-y0(2); x2-y0(2); x3-y0(2); x4-y0(2); x5-y0(2); x6-y0(2)])/Cov_X2*[x1-y0(2); x2-y0(2); x3-y0(2); x4-y0(2); x5-y0(2); x6-y0(2)])/2)) + p(3)* ((1/(sqrt(Deter_Cov_X3)*(2*pi)^(N/2)))*exp(-(transpose([x1-y0(3); x2-y0(3); x3-y0(3); x4-y0(3); x5-y0(3); x6-y0(3)])/Cov_X3*[x1-y0(3); x2-y0(3); x3-y0(3); x4-y0(3); x5-y0(3); x6-y0(3)])/2)));
    %Hy = integralN(@(x1,x2,x3,x4,x5,x6)arrayfun(Hy_gaussian_mixture,x1,x2,x3,x4,x5,x6),x1_min,x1_max,x1_min,x1_max,x1_min,x1_max,x1_min,x1_max,x1_min,x1_max,x1_min,x1_max);
    
    true_MI_dimN(1,iteration) = Hy - N/2-N/2*log(2*pi) - 1/2*sum(p.*log([Deter_Cov_X1 Deter_Cov_X2 Deter_Cov_X3]));
    true_MI_dimN(1,iteration) = true_MI_dimN(1,iteration) / log(2);
    

    for repo = 1:repos
        
        x_vec = zeros(1, samples);
        y_vec = x_vec;
        xy = zeros(length(p_array{1}), samples);
        ns = zeros(1, length(p_array{1}));
        
        x_dimN = zeros(1, samples);
        y_dimN = zeros(N, samples);
        xy_dimN = zeros(N*length(p_array{1}), samples);
        
        for n = 1:samples
            idxs = find(rand() < cumulative_p);
            x = idxs(1);
            x_vec(n) = x;
            y_vec(n) = sigma_y(x)*randn() + y0(x);            
            ns(x) = ns(x)+1;
            xy(x, ns(x)) = y_vec(n);
            
            x_dimN(n) = x;
            y_dimN(:,n) = mvnrnd(y0(x)*ones(1,N), (sigma_y(x))^2*eye(N),1);
            xy_dimN(((x-1)*N+1):(x*N),ns(x)) = y_dimN(:,n);
        end
    
        [one_dc_info, V] = ...
                discrete_continuous_info_fast(x_vec, y_vec, k, 2);
        estimated_MI(repo,iteration) = one_dc_info;
        [one_dc_info_dimN, V_dim10] = ...
                    discrete_continuous_info(x_dimN, y_dimN, k, 2);
        estimated_MI_dimN(repo,iteration) = one_dc_info_dimN;
    end
       
end
estimated_MI_cell = {estimated_MI,estimated_MI_dimN};
dims = [1,6];
for i = 1:length(estimated_MI_cell)
    dim = dims(i);
    mean_estimated_MI = mean(estimated_MI_cell{i}, 1);
    std_estimated_MI = std(estimated_MI_cell{i},0,1);
    [sort_mean_estimated_MI, sort_index] = sort(mean_estimated_MI);
    sort_std_estimated_MI = std_estimated_MI(sort_index);
    sort_true_MI = true_MI(sort_index);
    xaxis_index = 1:length(p_array);
    figure1 = plot(xaxis_index,sort_true_MI, 'r--o', xaxis_index, sort_mean_estimated_MI,'b');
    hold on
    xlim([0,16]);
    ylim([0, 2]);
    errorbar(xaxis_index,sort_mean_estimated_MI,sort_std_estimated_MI);
    title(sprintf('100 iterations, 10000 samples, discrete-dim=1, gaussian-dim=%d',4));
    legend({'true-MI','mean-estimated-MI'},'Location','northwest','Orientation','vertical');
    hold off
    saveas(figure1,sprintf('../result/Compare_estimatedMI_With_trueMI/gaussian/categorical/%diterations_%dsamples_discrete1_gaussian%d_MI.png',[100, 10000,dim]));
end




