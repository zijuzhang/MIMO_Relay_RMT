rows = 100;
cols = 100;
steps = 100;
bottom = .001;
top = 10; 
x_values = linspace(bottom, top , steps);
step_size = (top-bottom)/steps;
s_values = x_values + 1i*1e-6;
channel = rayleigh_channel(rows, cols, 1/sqrt(2*rows));
rho = .98;
correlation = exponential_correlation(rows, rho);
% check = correlation*correlation';
total_cov = (correlation*(channel*channel')*correlation');
stieltjes_values = (1./s_values).*(1+gamma_s(1./s_values));
% stieltjes_values = (1./s_values).*(1+gamma_s(s_values));
% stieltjes_values = marcenko_pastur(s_values,1);
pdf = 1/pi .* imag(stieltjes_values);
% pdf = imag(stieltjes_values);
num_capacity = MIMO_capacity(total_cov, 1/cols)
asymptotic_capacity  = aed_capacity(x_values, pdf, 1/cols, rows, step_size)
plot(x_values, pdf)
title(['Capacity for this PDF is: ', num2str(asymptotic_capacity)]);

function output = gamma_s(input)
    out = zeros(size(input), 'like', input);
    for i = 1:length(input)
        out(i) =  gamma_s_func(input(i));
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

function x = gamma_s_func(eval_point)
init_point = rand() + 1j*rand();
x = fsolve(@root1,init_point);
    function F = root1(z)
        F = z - eval_point.*(1+z)./S_func(z);
%         F = z - eval_point*power((1+z),2);
    end
end

function val = S_func(z)
%     val = S_from_inv_gamma(z, evals) % Includes quarter circle
%     val = 1./power((1+z),2)
%     wishart = 1./(1+z);
    rho = .98;
    alpha = (1+rho^2)/(1-rho^2);
%     correlation = (alpha*z + sqrt(alpha^2*z^2 - (z^2 - 1)))/(z-1);
%     val = wishart*correlation;
%     val = 1/((z+1)*(1+z));
    val = (alpha*z + sqrt(alpha^2*z^2 - (z^2 - 1)))/((z-1)*(1+z)*(1+z));
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

