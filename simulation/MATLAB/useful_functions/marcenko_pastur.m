function [outputArg1] = marcenko_pastur(s, beta)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
outputArg1 = 1/2 + (1-beta)./(2*s) + sqrt(power(1-beta, 2)./(4*power(s, 2))-(1+beta)./(2*s)+1/4);
% outputArg1 = sqrt(power(1-beta, 2)./(4*power(s, 2))-(1+beta)./(2*s)+1/4) - 1/2 - (1-beta)./(2*s);
end

