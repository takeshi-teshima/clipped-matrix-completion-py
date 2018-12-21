function retVal=h_derivative(sol_M, C, lambda3,U,S,V)
W = sol_M >= C;

num_keep=nnz(diag(S)>1e-8);

retVal = lambda3 * ((1 - W) .* (U(:,1:num_keep)*V(:,1:num_keep)'));