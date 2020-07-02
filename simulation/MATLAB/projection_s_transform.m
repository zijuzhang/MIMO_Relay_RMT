rows = 100;
cols = 100;
steps = 100;
bottom = 1e-3;
top = 10; 
x_values = linspace(bottom, top , steps);
step_size = (top-bottom)/steps;
s_values = x_values + 1i*1e-3;
betas = linspace(.1, 1, 5);
numeric_capacity = zeros(size(betas));
asymptotic_capacity = zeros(size(betas));
for i = 1:length(betas)
    numeric_capacity(i) = average_numeric_capacity(rows, cols, betas(i), 200);
    stieltjes_values = (1./s_values).*(1+gamma_s(1./s_values, betas(i)));
    pdf = 1/pi .* imag(stieltjes_values);
    pdf = abs(pdf);
    asymptotic_capacity(i)  = aed_capacity(x_values, pdf, 1/cols, rows, step_size);
end
numeric_capacity
plot(betas, numeric_capacity, '-s');
hold on
plot(betas, asymptotic_capacity, '-o');
legend('numeric capacity','asymptotic capacity');
title('Numeric Vs. Asyptotic Correlated Capacity');
xlabel('Projector \beta');
ylabel('Capacity (bits)');
f = gca;
exportgraphics(f,'results\projection.png')
clear all;

function output = gamma_s(input, beta)
    out = zeros(size(input), 'like', input);
    for i = 1:length(input)
        out(i) =  gamma_s_func(input(i), beta);
    end
    output = out;
end

function x = gamma_s_func(eval_point, beta)
    function F = root1(z)
        F = z.*S_func(z, beta) - eval_point - z*eval_point;
    end
init_point = rand() + 1j*rand();
x = fsolve(@root1,init_point);
end

function val = S_func(z, beta)
    wishart = 1./power((1+z),1);
    projection = (1+z)/(z+beta);
    val = projection*wishart*projection*wishart;
end

function ave = average_numeric_capacity(rows, cols, beta, average)
    vals = zeros(average,1);
    for i = 1:length(vals)
%         awgn = rayleigh_channel(rows, cols, 1/sqrt(2*rows));
        awgn = rayleigh_channel(rows, cols, 1/sqrt(2*(rows*beta)));
        projector1 = projection_matrix(rows, beta);
        projector2 = projection_matrix(rows, beta);
%         correlation = exponential_correlation(rows, rho);
%         total_channel = channel1;
%         total_channel = channel2*channel1;
%         total_channel = correlation*channel2*correlation*channel1*correlation;
        total_channel = projector1*awgn;
%         total_channel = projector2*awgn*projector1*awgn;
        total_cov = (total_channel*total_channel');
        vals(i) = MIMO_capacity(total_cov, 1/cols);
    end
    ave = mean(vals);
end