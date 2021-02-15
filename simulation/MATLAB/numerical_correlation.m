rows = 100;
cols = 100;

rhos = linspace(0, 1.0, 10);
numeric_capacity = zeros(size(rhos));

for i = 1:length(rhos)
    numeric_capacity(i) = average_numeric_capacity(rows, cols, rhos(i), 1000);
end

figure(2)
plot(rhos, numeric_capacity, '-s');
legend('numeric capacity','asymptotic capacity');
title('Numeric Vs. Asyptotic Correlated Capacity');
xlabel('Exponential Correlation Value (rho)');
ylabel('Capacity (bits)');
f = gca;
exportgraphics(f,'results\comparison.png')
clear all;

function ave = average_numeric_capacity(rows, cols, rho, average)
    vals = zeros(average,1);
    for i = 1:length(vals)
        channel1 = rayleigh_channel(rows, cols, 1/sqrt(2*rows));
        channel2 = rayleigh_channel(rows, cols, 1/sqrt(2*rows));
        correlation = exponential_correlation(rows, rho);
%         total_channel = channel1;
%         total_channel = channel2*channel1;
        total_channel = correlation*channel2*channel1*correlation;
%         total_channel = correlation*channel2*correlation*channel1*correlation;
        total_cov = (total_channel*total_channel');
        vals(i) = MIMO_capacity(total_cov, 1/cols);
    end
    ave = mean(vals);
end