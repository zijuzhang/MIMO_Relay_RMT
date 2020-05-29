rows = 100;
cols = 100;
steps = 100;
bottom = .01;
top = 10; 
channel = rayleigh_channel(rows, cols, 1/sqrt(2*rows));
block_size = 2;
block = toeplitz(ones(1,rows/block_size));
correlation = kron(eye(block_size), block);
correlation = normr_la(correlation);
% correlation = correlation*correlation';
total_cov = (correlation*(channel*channel')*correlation');
det_eigen_values = eig(correlation);
x_values = linspace(bottom, top , steps);
step_size = (top-bottom)/steps;
s_values = x_values + 1i*1e-4;
stieltjes_values = (1./s_values).*(1+gamma_s(1./s_values, det_eigen_values));
pdf = 1/pi .* imag(stieltjes_values);
num_capacity = MIMO_capacity(total_cov, 1/cols)
asymptotic_capacity  = aed_capacity(x_values, pdf, 1/cols, rows, step_size)
plot(x_values, pdf)
title(['Capacity for this PDF is: ', num2str(asymptotic_capacity)])

function output = gamma_s(input, eval)
    out = zeros(size(input), 'like', input);
    for i = 1:length(input)
        init_point = rand() + 1j*rand();
        out(i) =  gamma_s_func(input(i), init_point, eval);
    end
    output = out;
end


function x = gamma_s_func(eval_point, init_point, evals)
x = fsolve(@root2d,init_point);
      function F = root2d(gamma_z)
%             F = gamma_z - eval_point.*power(1+gamma_z,2);
            F = gamma_z - eval_point.*(1+gamma_z)./S_func(gamma_z, evals);
      end
end


function val = S_func(z, evals)
    val = S_from_inv_gamma(z, evals)./(1+z);
%     val = 1./(1+z);
end

function val = S_from_inv_gamma(z, evals)
    val = inv_gamma(z, evals).*(z+1)./z
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
        val = val + (1/(det_eigen_values(i)-s))/N;
    end
end

function val = stieljes_qc(s)
    val = (1/2).*sqrt(1-4/s) - 1/2;
end

