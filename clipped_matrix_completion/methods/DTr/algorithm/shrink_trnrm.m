function [sol_M trnrm]=shrink_trnrm(svd_obj, eta_t, lambda1, epsilon)

[U,S,V]=svdecon(svd_obj);
S=diag(S)-(eta_t*lambda1);
num_keep=nnz(S>epsilon);

ret_U = U(:,1:num_keep);
ret_S = S(1:num_keep);
ret_V = V(:,1:num_keep);
sol_M=ret_U * diag(ret_S) * ret_V';
trnrm = sum(ret_S);