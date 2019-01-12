function [retVal,h_U,h_S,h_V]=h_function(sol_M, C, lambda2)
W = sol_M >= C;

% svd_obj=W.*(sol_M-C);
svd_obj=sol_M - W.*(sol_M - C);

[h_U,h_S,h_V]=svdecon(svd_obj);

retVal=lambda2*sum(diag(h_S));