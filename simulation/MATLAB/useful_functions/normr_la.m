function ret_mat = normr_la(matrix)
    ret_mat = matrix;
    rows = size(matrix,2);
    for i = 1 : rows
    	ret_mat(i,:) = matrix(i,:)./norm(matrix(i,:));
    end
end
