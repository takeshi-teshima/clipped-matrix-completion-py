function sol_M=clipping_aware_matrix_completion(M_path, Omega_M_path, C, ...
                                                lambda1, lambda2, lambda3, T, ...
                                                eta_t, decay_rate, clip_or_hinge,initialization)

    f_function = @f_function_hinge;
    f_derivative = @f_derivative_hinge;

    % M_path contains M
    C = double(C);
    eta_t = double(eta_t);
    lambda1 = double(lambda1);
    lambda2 = double(lambda2);
    lambda3 = double(lambda3);
    decay_rate = double(decay_rate);

    load(M_path);
    load(Omega_M_path);

    [n, d] = size(M);
    rng(1);
    if (strcmp(initialization, 'zeros'))
        sol_M=zeros(n,d);
    elseif(strcmp(initialization, 'ones'))
        sol_M= ones(n,d);
    elseif(strcmp(initialization, 'large'))
        sol_M=C * ones(n,d);
    end

    R_C = (M == C);

    [h_ret,h_U,h_S,h_V]=h_function(sol_M, C,lambda3);

    for i=1:T
        if (mod(i,10)==0)
            disp(["Round: " num2str(i)]);
        end
        subgradient=f_derivative(sol_M, C, M,Omega_M,lambda1, R_C)+h_derivative(sol_M, C,lambda3,h_U,h_S,h_V);
        svd_obj=sol_M-eta_t*subgradient;
        [U,S,V]=svdecon(svd_obj);
        S=diag(S)-(eta_t*lambda2);
        num_keep=nnz(S>1e-8);

        sol_M=U(:,1:num_keep)*diag(S(1:num_keep))*V(:,1:num_keep)';

        [h_ret,h_U,h_S,h_V]=h_function(sol_M, C, lambda3);
        tr_X = lambda2*sum(S(1:num_keep));
        f_value = f_function(sol_M, C, M,Omega_M,lambda1, R_C);
        obj=tr_X+f_value+h_ret;

        fprintf('%f = %f + %f + %f', obj, tr_X, f_value, h_ret);
        fprintf('\n');

        if eta_t>40
            eta_t=eta_t*decay_rate;
        else
            eta_t=eta_t;
        end
    end

    disp(["Fine"]);