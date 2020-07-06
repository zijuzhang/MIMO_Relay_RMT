 function matrix = rayleigh_channel(rows, cols, var)
	matrix = normrnd(0, var, rows, cols) + 1i*normrnd(0, var, rows, cols);
%     matrix = normrnd(0, var, rows, cols);
 end
