rows = 100;
cols = 100;
steps = 200;
bottom = 1e-3;
% top = 200; 
top = 20; 
x_values = linspace(bottom, top , steps);
step_size = (top-bottom)/steps;
s_values = x_values + 1i*1e-3;
% s_values = x_values;
rhos = linspace(0, 0.9, 10);
% rhos = linspace(0,1-1e-4,10);
numeric_capacity = zeros(size(rhos));
asymptotic_capacity = zeros(size(rhos));
% figure(1)
for i = 1:length(rhos)
    numeric_capacity(i) = average_numeric_capacity(rows, cols, rhos(i), 100);
    stieltjes_values = (1./s_values).*(1+gamma_s(1./s_values, rhos(i)));
%     stieltjes_values = (1./s_values).*(1+gamma_s(s_values, rhos(i)));
    pdf = 1/pi .* imag(stieltjes_values);
%     pdf = abs(pdf);
    asymptotic_capacity(i)  = aed_capacity(x_values, pdf, 1/cols, rows, step_size);
%     plot(x_values,pdf);
%     title(['Capacity for this PDF is: ', num2str(asymptotic_capacity)]);
%     hold on
end
numeric_capacity
asymptotic_capacity
figure(2)
plot(rhos, numeric_capacity, '-s');
hold on
plot(rhos, asymptotic_capacity, '-o');
legend('numeric capacity','asymptotic capacity');
title('Numeric Vs. Asyptotic Correlated Capacity');
xlabel('Exponential Correlation Value (rho)');
ylabel('Capacity (bits)');
f = gca;
exportgraphics(f,'results\RxCorrelation.png')
clear all;

function output = gamma_s(input, rho)
    out = zeros(size(input), 'like', input);
    for i = 1:length(input)
        out(i) =  gamma_s_func(input(i), rho);
%         out(i) =  new(input(i), rho);
    end
    output = out;
end

function x = new(eval_point, rho)
alpha = (1+rho^2)/(1-rho^2);
beta = 1; %Nt/Nr
    function F = root1(z)
        F = (z+beta)^2*(z^2-1)/(eval_point^2) - z^2*(2*alpha*(z+beta)/eval_point - 1);
    end
init_point = rand() + 1j*rand();
x = fsolve(@root1,init_point);
x = real(x) + 1j*abs(imag(x));
end

function x = gamma_s_func(eval_point, rho)
init_point = rand() + 1j*rand();
% options = optimset('notify');
    function F = root1(z)
        %Correct negative imaginary component
        F = z.*S_func(z, rho) - eval_point - z*eval_point; % more consistent
%         F =  eval_point - S_func(z, rho)*z/(1+z);
    end
x = fsolve(@root1,init_point); %TODO reject negative values
% x = real(x) + 1j*abs(imag(x));
end

function val = S_func(z, rho)
    wishart = 1./power((1+z),2);
%     wishart = 1./power((1+z),1);
    alpha = (1+rho^2)/(1-rho^2);
    correlation = (alpha*z - sqrt(alpha^2*z^2 - (z^2 - 1)))/(z-1);
%     correlation = (alpha*z + sqrt(alpha^2*z^2 - (z^2 - 1)))/(z-1);
%     val = wishart*correlation;
    val = wishart*correlation*correlation
%     val = wishart*correlation*correlation*correlation;
%     val = 1./power((1+z),2);
end


function ave = average_numeric_capacity(rows, cols, rho, average)
    vals = zeros(average,1);
    for i = 1:length(vals)
        channel1 = rayleigh_channel(rows, cols, 1/sqrt(2*rows));
        channel2 = rayleigh_channel(rows, cols, 1/sqrt(2*rows));
        correlation = exponential_correlation(rows, rho);
%         total_channel = channel1;
%         total_channel = channel2*channel1;
%         total_channel = correlation*channel1;
%         total_channel = correlation*channel2*channel1;
        total_channel = correlation*channel2*correlation*channel1;
%         total_channel = correlation*channelx2*correlation*channel1*correlation;
%         total_channel = correlation*projector2*channel2*correlation*projector1*channel1*correlation;
        total_cov = (total_channel*total_channel');
        vals(i) = MIMO_capacity(total_cov, 1/cols);
    end
    ave = mean(vals);
end