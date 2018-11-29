function retVal=h_subgrad(X, C,U,S,V)
W = X < C;

num_keep=nnz(diag(S)>1e-8);

retVal = W .* (U(:,1:num_keep)*V(:,1:num_keep)');