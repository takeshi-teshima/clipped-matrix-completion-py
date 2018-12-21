function retVal=f_derivative_hinge(sol_M, C, M,Omega_M,lambda1, R_C)

% This one is outside the theory
retVal=2*lambda1*((sol_M-M).*Omega_M .* (1 - R_C) - max(M - sol_M, 0).*Omega_M ...
                  .* R_C);