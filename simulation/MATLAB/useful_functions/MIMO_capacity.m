 function capacity = MIMO_capacity(covariance_matrix, snr)
 capacity = sum(log2(1 + snr.*eig(covariance_matrix)));
 end
