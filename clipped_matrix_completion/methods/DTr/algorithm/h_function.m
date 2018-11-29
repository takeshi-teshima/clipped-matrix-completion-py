function [retVal,h_U,h_S,h_V]=h_function(X, C)
%% $W^{(t)} = 1\{X^{(t)} < C\}$
W = X < C;

%% $svd_obj(\X) = X \odot W^{(t)} + C (1 - W^{(t)})$
svd_obj=X.*W + C * (1 - W);
[h_U,h_S,h_V]=svdecon(svd_obj);

retVal=sum(diag(h_S));