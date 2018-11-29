function retVal=f_function_sqhinge(sol_M, C, M,Omega_M, R_C)

a = sol_M-M;
retVal=norm(Omega_M.* (a + R_C .* (max(M - sol_M, 0) - a)), 'fro')^2;