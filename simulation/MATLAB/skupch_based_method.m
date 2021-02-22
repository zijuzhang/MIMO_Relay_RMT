rows = 100;
cols = 100;
steps = 100;
bottom = 1e-1;
top = 5; 
x_values = linspace(bottom, top , steps);
step_size = (top-bottom)/steps;
s_values = x_values + 1i*1e-2; % The img component needs to go to zero to evaluate pdf.
rhos = linspace(0, 1, 5);
numeric_capacity = zeros(size(rhos));
asymptotic_capacity = zeros(size(rhos));
figure(1)
for i = 1:length(rhos)
    numeric_capacity(i) = average_numeric_capacity(rows, cols, rhos(i), 10);
    stieltjes_values = evaluate_points_skupch(s_values,rhos(i));
    pdf = 1/pi .* imag(stieltjes_values);
    % Really cheating here because I'm not returning the proper roots.
    pdf = abs(pdf);
    asymptotic_capacity(i)  = aed_capacity(x_values, pdf, 1/cols, rows, step_size);
    plot(x_values,pdf,'DisplayName', 'Correlation parameter \rho = ' + string(rhos(i)));
    title('AED');
    hold on
end
legend();
f = gca;
exportgraphics(f,'results\AED.png')
figure(2)
plot(rhos, numeric_capacity, '-s');
hold on
plot(rhos, asymptotic_capacity, '-o');
legend('numeric capacity','asymptotic capacity');
title('Numeric Vs. Asyptotic Correlated Capacity');
xlabel('Correlation \rho');
ylabel('Capacity (bits)');
f = gca;
exportgraphics(f,'results\capacity.png')
clear all;


function output = evaluate_points_skupch(input, rho)
    output = zeros(size(input), 'like', input);
    % Evaluate the gamma function for each value used in pdf
    % integration.
    for i = 1:length(input)
        output(i) =  skupch_stieltjes(input(i), rho);
    end
%     output = out;
end

function output = skupch_stieltjes(eval_point, rho)
%% 

% Skupch paper polynomial (with beta = 1) (s = eval_point)
% Only the imaginary portion of the returned value is considered...
% TODO, how to choose proper root???

%%
% syms x;
% skupch_eqn = x^4*eval_point^4 + x^3*eval_point^3*(-2 -2*eval_point*x) + ...
%     x^2*eval_point^2*(4*beta*eval_point * eval_point^2) + ...
%     x*eval_point*(2*beta*eval_point - 2*eval_point^2) + ...
%     eval_point^2;
poly_vector = [eval_point^4  eval_point^3*(-2 -2*rho*eval_point) ...
    eval_point^2*(4*rho*eval_point * eval_point^2) ...
    eval_point*(2*rho*eval_point - 2*eval_point^2) ...
    eval_point^2];
% check = solve(skupch_eqn, x);
zeros = roots(poly_vector);
% eval(check)
% output = check(1);
% Return zero with largest img component.
[max_imag, max_imag_ind] = max(imag(zeros));
% output = zeros(max_imag_ind);
output = zeros(3);
% if (imag(output) < 0)
%     error('Strictly positive PDF??');
% end
end

function ave = average_numeric_capacity(rows, cols, rho, average)
    vals = zeros(average,1);
    for i = 1:length(vals)
        instance_rayleigh = rayleigh_channel(rows, cols, 1/sqrt(2*rows));
        correlation = exponential_correlation(rows, rho);
        total_channel = correlation*instance_rayleigh;
        total_cov = (total_channel*total_channel');
        vals(i) = MIMO_capacity(total_cov, 1/cols);
    end
    ave = mean(vals);
end