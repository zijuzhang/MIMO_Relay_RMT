function [matrix] = exponential_correlation(dim, rho)
%EXPONENTIAL_CORRELATION Summary of this function goes here
%  Just need to find first row of the matrix
vec = ones(1,dim);
for i = 1 : dim
    vec(i) = rho^(i-1);
end
covariance = toeplitz(vec);
matrix= sqrtm(covariance);
end

