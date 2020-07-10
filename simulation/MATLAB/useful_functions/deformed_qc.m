function [outputArg1] = deformed_qc(s, alpha)
%DEFORMED_QC Summary of this function goes here
%   Detailed explanation goes here
outputArg1 = sqrt((1-alpha).^2./(4.*s.^2) - (1+alpha)./(2.*s) + 1/4) - 1/2 - (1-alpha)./(2.*s);
% outputArg1 = sqrt((1-alpha).^2./(4.*s.^2) - (1+alpha)./(2.*s) + 1/4) + 1/2 + (1-alpha)./(2.*s);
end 

