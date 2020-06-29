rows = 100;
cols = 100;
steps = 100;
bottom = .001;
top = 10; 
x_values = linspace(bottom, top , steps);
step_size = (top-bottom)/steps;
s_values = x_values + 1i*1e-6;
rhos = linspace(.5,.99,5);
numeric_capacity = zeros(size(rhos));
asymptotic_capacity = zeros(size(rhos));
for i = 1:length(rhos)
    stieltjes_values = (1./s_values).*(1+gamma_s(1./s_values, rhos(i)));
    % stieltjes_values = (1./s_values).*(1+gamma_s(s_values));
    % stieltjes_values = marcenko_pastur(s_values,1);
    pdf = 1/pi .* imag(stieltjes_values);
    % pdf = imag(stieltjes_values);
    numeric_capacity(i) = average_numeric_capacity(rows, cols, rhos(i), 20);
    asymptotic_capacity(i)  = aed_capacity(x_values, pdf, 1/cols, rows, step_size);
%     plot(x_values, pdf)
%     title(['Capacity for this PDF is: ', num2str(asymptotic_capacity)]);
end

plot(rhos, numeric_capacity, '-s')
hold on
plot(rhos, asymptotic_capacity, '-o')
legend('numeric_capacity','asymptotic_capacity')
title('Numeric Vs. Asyptotic Correlated Capacity');





function output = gamma_s(input, rho)
    out = zeros(size(input), 'like', input);
    for i = 1:length(input)
        out(i) =  gamma_s_func(input(i), rho);
%         out(i) =  new(input(i));
    end
    output = out;
end


function x = new(eval_point)
init_point = rand() + 1j*abs(rand());
rho = .1;
alpha = (1+rho^2)/(1-rho^2);
beta =1;
x = fsolve(@root1,init_point);
    function F = root1(z)
%         F = eval_point*(1+z)/S_func(z) - z;
        F = (z+beta)^2*(z^2-1)/(eval_point^2) - z^2*(2*alpha*(z+beta)/eval_point - 1);
    end
end

function x = gamma_s_func(eval_point, rho)
init_point = rand() + 1j*rand();
x = fsolve(@root1,init_point);
    function F = root1(z)
        F = z.*S_func(z, rho) - eval_point - z*eval_point;
%         F = z - eval_point.*(1+z)./S_func(z);
%         F = z - eval_point*power((1+z),2);
    end
end

function val = S_func(z, rho)
%     val = S_from_inv_gamma(z, evals) % Includes quarter circle
    wishart = 1./power((1+z),2);
    alpha = (1+rho^2)/(1-rho^2);
    correlation = (alpha*z - sqrt(alpha^2*z^2 - (z^2 - 1)))/(z-1);
    val = wishart*correlation;
%     val = 1./power((1+z),2)
end


function val = stieljes_corr(s, det_eigen_values)
    val = 0;
    N = length(det_eigen_values);
    for i = 1 : N
%         val = val + (1/(det_eigen_values(i)-s))/N;
        val = val + (1/(s-det_eigen_values(i)))/N;
    end
end

function val = stieljes_qc(s)
    val = (1/2).*sqrt(1-4/s) - 1/2;
end


function ave = average_numeric_capacity(rows, cols, rho, average)
    vals = zeros(average,1);
    for i = 1:length(vals)
        channel1 = rayleigh_channel(rows, cols, 1/sqrt(2*rows));
        channel2 = rayleigh_channel(rows, cols, 1/sqrt(2*rows));
        correlation = exponential_correlation(rows, rho);
        total_cov = (channel2*(correlation*(channel1*channel1')*correlation')*channel2');
%         total_cov = (correlation*(channel*channel')*correlation');
        vals(i) = MIMO_capacity(total_cov, 1/cols);
    end
    ave = mean(vals);
end

