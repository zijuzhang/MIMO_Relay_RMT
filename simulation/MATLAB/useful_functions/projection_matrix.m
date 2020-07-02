function ret_mat = projection_matrix(size, beta)
    vec = ones(1,size);
    vec(round(size*(beta)):end) = 0;
    ret_mat = diag(vec);
end