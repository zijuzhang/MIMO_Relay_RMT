function ret_mat = projection_matrix(size, beta)
    vec = ones(1,size);
    vec(max(1,round(size*(beta))):end) = 0;
    ret_mat = diag(vec);
end