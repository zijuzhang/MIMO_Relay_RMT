rows = 100;
cols = 100;
steps = 1000;
bottom = 1e-3;
top = 10; 
x_values = linspace(bottom, top , steps);
step_size = (top-bottom)/steps;
s_values = x_values + 1i*1e-3;
betas = linspace(.1, 1, 8);
numeric_capacity = zeros(size(betas));
asymptotic_capacity = zeros(size(betas));
for i = 1:length(betas)
    numeric_capacity(i) = average_numeric_capacity(rows, cols, betas(i), 100);
end
numeric_capacity
asymptotic_capacity
figure(2)
plot(betas, numeric_capacity, '-s');
% hold on
% plot(betas, asymptotic_capacity, '-o');
legend('numeric capacity','asymptotic capacity');
title('Numeric Vs. Asyptotic Correlated Capacity');
xlabel('Exponential Correlation Value (rho)');
ylabel('Capacity (bits)');
f = gca;
exportgraphics(f,'results\comparison.png')
clear all;

function ave = average_numeric_capacity(rows, cols, beta, average)
    vals = zeros(average,1);
    for i = 1:length(vals)
        channel1 = rayleigh_channel(rows, cols, 1/sqrt(2*rows));
        channel2 = rayleigh_channel(rows, cols, 1/sqrt(2*rows));
        projector1 = projection_matrix(rows, beta);
        projector2 = projection_matrix(rows, beta);
%         correlation = exponential_correlation(rows, rho);
%         total_channel = channel1;
%         total_channel = channel2*channel1;
%         total_channel = correlation*channel2*correlation*channel1*correlation;
        total_channel = projector2*channel2*projector1*channel1;
        total_cov = (total_channel*total_channel');
        vals(i) = MIMO_capacity(total_cov, 1/cols);
    end
    ave = mean(vals);
end