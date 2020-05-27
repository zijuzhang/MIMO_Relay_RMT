function x = gamma_s_func(eval_point, init_point)
x = fsolve(@root2d,init_point);
      function F = root2d(gamma_z)
%             F = gamma_z - eval_point.*power(1+gamma_z,2);
            F = gamma_z - eval_point.*(1+gamma_z)./S_func(gamma_z);
      end
end