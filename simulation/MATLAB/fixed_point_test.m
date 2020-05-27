rows = 100;
cols = 100;
steps = 100;
bottom = .01;
top = 10; 
block = toeplitz([1, .5]);
correlation = kron(eye(cols/2), block)
det_eigen_values = eig(correlation);
x_values = linspace(bottom, top , steps);
step_size = (top-bottom)/steps;
s_values = x_values + 1i*1e-4;
stieltjes_values = (1./s_values).*(1+gamma_s(1./s_values));
pdf = 1/pi .* imag(stieltjes_values);
asymptotic_capacity  = aed_capacity(x_values, pdf, 1/cols, rows, step_size)
plot(x_values, pdf)