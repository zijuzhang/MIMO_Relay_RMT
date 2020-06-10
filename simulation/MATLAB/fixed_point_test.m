rows = 100;
cols = 100;
steps = 100;
bottom = .001;
top = 10; 
channel = rayleigh_channel(rows, cols, 1/sqrt(2*rows));
rho = .9;
correlation = exponential_correlation(rows, rho);
total_cov = (correlation*(channel*channel')*correlation');
x_values = linspace(bottom, top , steps);
step_size = (top-bottom)/steps;
s_values = x_values + 1i*1e-6;
% stieltjes_values = (1./s_values).*(1+gamma_s(s_values));
stieltjes_values = marcenko_pastur(s_values,1);
pdf = 1/pi .* imag(stieltjes_values);
num_capacity = MIMO_capacity(total_cov, 1/cols)
asymptotic_capacity  = aed_capacity(x_values, pdf, 1/cols, rows, step_size)
plot(x_values, pdf)
title(['Capacity for this PDF is: ', num2str(asymptotic_capacity)]);

function output = gamma_s(input)
    out = zeros(size(input), 'like', input);
    for i = 1:length(input)
%         out(i) =  gamma_s_func(input(i), eval);
        out(i) =  new(input(i));
    end
    output = out;
end

function x = inverse_gamma_cor(eval_point)
    rho = .9;
    alpha = (1+rho^2)/(1-rho^2);
    val = (alpha*z^2 + z*sqrt(alpha^2*z^2 - (z^2 -1)))/((z^2-1)*(z+1))
end

function x = new(eval_point)
init_point = rand() + .1j*rand();
rho = .5;
alpha = (1+rho^2)/(1-rho^2);
x = fsolve(@root1,init_point);
    function F = root1(gamma_inv_s)
        F = ((gamma_inv_s + 1)^2*(gamma_inv_s^2 - 1))/(eval_point^2) ...
            - gamma_inv_s^2*(2*alpha*(gamma_inv_s + 1)/eval_point - 1);
    end
end

function x = gamma_s_func(eval_point, evals)
init_point = rand() + 1j*rand();
x = fsolve(@root2d,init_point);
      function F = root2d(gamma_z)
%             F = gamma_z - eval_point.*power(1+gamma_z,2); %   works
            F = gamma_z - eval_point.*(1+gamma_z)./S_func(gamma_z, evals);
      end
end


function val = S_func(z, evals)
%     val = S_from_inv_gamma(z, evals) % Includes quarter circle
%     val = 1./(1+z)
    wishart = 1./(1+z);
    rho = .9;
    alpha = (1+rho^2)/(1-rho^2);
    correlation = (alpha*z + sqrt(alpha^2*z^2 - (z^2 - 1)))/((z-1)*(1+z))
    val = wishart*correlation;
%     val = (alpha*z + sqrt(alpha^2*z^2 - (z^2 - 1)))/((z-1)*(1+z));
end

function val = S_from_inv_gamma(z, evals)
    val = inv_gamma(z, evals).*(z+1)./z;
end

function val = inv_gamma(eval_point, evals)
init_point = rand() + 1j*rand();
val = fsolve(@root,init_point);
      function F = root(gamma_z)
%             F = gamma_z.*(eval_point+1) + stieljes_corr(1/gamma_z, evals);
            F = gamma_z.*(eval_point+1) + stieljes_qc(1/gamma_z);
      end
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

