function output = gamma_s(input)
    out = zeros(size(input), 'like', input);
    for i = 1:size(input)
        init_point = rand() + 1j*rand();
        out(i) =  gamma_s_func(input(i), init_point);
    end
    output = out;
end
