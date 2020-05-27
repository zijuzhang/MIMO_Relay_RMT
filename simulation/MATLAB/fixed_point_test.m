x_values = linspace(.01, 10, 100);
s_values = x_values + 1j*1e-6;
stieltjes_values = (1./s_values).*(1+gamma_s(1./s_values))
pdf = 1/pi .* imag(stieltjes_values);
plot(x_values, stieltjes_values)