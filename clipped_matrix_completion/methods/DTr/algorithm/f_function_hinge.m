function retVal=f_function_hinge(sol_M, C, M,Omega_M,lambda1, R_C)

% This one is outside the theory
loss_below_clip = norm(Omega_M.* (1 - R_C) .* (sol_M-M),'fro')^2;
loss_above_clip = norm(Omega_M.* R_C .* max(M - sol_M, 0),'fro')^2;
retVal=lambda1*(loss_below_clip + loss_above_clip);