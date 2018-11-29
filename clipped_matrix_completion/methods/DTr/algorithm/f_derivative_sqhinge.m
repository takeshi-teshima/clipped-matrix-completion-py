function retVal=f_derivative_sqhinge(sol_M, C, M,Omega_M, R_C)
a = sol_M-M;
retVal=2*((a - (a + max(M - sol_M, 0)) .* R_C).*Omega_M);