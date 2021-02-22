rows = 100;
cols = 100;
steps = 100;
bottom = 1e-1;
top = 7; 
x_values = linspace(bottom, top , steps);
step_size = (top-bottom)/steps;
s_values = x_values + 1i*1e-7; % The img component needs to go to zero to evaluate pdf.
betas = linspace(0.3, 1, 10);
numeric_capacity = zeros(size(betas));
asymptotic_capacity = zeros(size(betas));
figure(1)
for i = 1:length(betas)
    numeric_capacity(i) = average_numeric_capacity(rows, cols, betas(i), 10);
%     stieltjes_values = (1./s_values).*(1+gamma_s(1./s_values, betas(i)));
    stieltjes_values = evaluate_points_skupch(s_values,betas(i));
    pdf = 1/pi .* imag(stieltjes_values);
    % Really cheating here because I'm not returning the proper roots.
    pdf = abs(pdf);
    asymptotic_capacity(i)  = aed_capacity(x_values, pdf, 1/cols, rows, step_size);
    plot(x_values,pdf,'DisplayName', '\beta = ' + string(betas(i)));
    title('AED');
    hold on
end
legend();
f = gca;
exportgraphics(f,'results\AED.png')
figure(2)
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


function output = evaluate_points_skupch(input, beta)
    output = zeros(size(input), 'like', input);
    % Evaluate the gamma function for each value used in pdf
    % integration.
    for i = 1:length(input)
        output(i) =  skupch_stieltjes(input(i), beta);
    end
%     output = out;
end

function output = skupch_stieltjes(eval_point, beta)
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
poly_vector = [eval_point^4  eval_point^3*(-2 -2*beta*eval_point) ...
    eval_point^2*(4*beta*eval_point * eval_point^2) ...
    eval_point*(2*beta*eval_point - 2*eval_point^2) ...
    eval_point^2];
% check = solve(skupch_eqn, x);
zeros = roots(poly_vector);
% eval(check)
% output = check(1);
% Return zero with largest img component.
% [max_imag, max_imag_ind] = max(imag(zeros));
% output = zeros(max_imag_ind);
output = zeros(3);
if (imag(output) < 0)
%     error('Strictly positive PDF??');
end
end



function output = gamma_s(input, beta)
    output = zeros(size(input), 'like', input);
    % Evaluate the gamma function for each value used in pdf
    % integration.
    for i = 1:length(input)
        output(i) =  gamma_s_func(input(i), beta);
    end
%     output = out;
end



function x = gamma_s_func(eval_point, beta)
    function F = root1(z)
        F = z.*S_func(z, beta) - eval_point - z*eval_point;
    end
% init_point = rand() + 1j*rand();
% x = fsolve(@root1,init_point);
% sq_roots = [2, 0, 1]
% sq_check = roots(sq_roots);
% polynomial_vector = [eval_point, (2*eval_point-1), eval_point];
% check1 = roots(polynomial_vector)
syms x;
eqn = x^2*eval_point + x*(2*eval_point-1) + eval_point == 0;




check = solve(skupch_eqn, x);
eval(check)
% pos_img = imag(check(1).val)>0
%   Need to pick out positive part of the polynomial
x = check(1);
end

function val = S_func(z, beta)
%     wishart = 1./power((1+z),1);
    wishart = 1./(1+z); 
    projection = (1+z)/(z+beta);
%     val = projection*wishart*projection*wishart;
%     val = wishart*projection;
    val = wishart;
end

function ave = average_numeric_capacity(rows, cols, beta, average)
    vals = zeros(average,1);
    for i = 1:length(vals)
        instance_rayleigh = rayleigh_channel(rows, cols, 1/sqrt(2*rows));
%         awgn = rayleigh_channel(rows, cols, 1/sqrt(2*(rows*beta)));
%         projector1 = projection_matrix(rows, beta);
%         projector2 = projection_matrix(rows, beta);
        correlation = exponential_correlation(rows, rho);
        total_channel = correlation*instance_rayleigh;
        total_cov = (total_channel*total_channel');
        vals(i) = MIMO_capacity(total_cov, 1/cols);
    end
    ave = mean(vals);
end