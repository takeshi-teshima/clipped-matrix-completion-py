function retVal=h_derivative(sol_M, C, lambda2,U,S,V)
W = sol_M >= C;

num_keep=nnz(diag(S)>1e-8);

retVal = lambda2 * ((1 - W) .* (U(:,1:num_keep)*V(:,1:num_keep)'));