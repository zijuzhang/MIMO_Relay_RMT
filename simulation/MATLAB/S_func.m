function val = S_func(z)
    val = S_from_inv_gamma(z)./(1+z);
end

function val = S_from_inv_gamma(z)
    val = inv_gamma(z).*(z+1).*z
end

function val = inv_gamma(eval_point)
init_point = rand() + 1j*rand();
val = fsolve(@root,init_point);
      function F = root(gamma_z)
            F = gamma_z + stieljes_corr(1/gamma_z)./(eval_point+1);
      end
end

function val = stieljes_corr(s)
    N = length(det_eigen_values);
    for i = 1 : N
        val = val + (1/(det_eigen_values(i)-s))/N;
    end
end
