function sol_M=clipping_aware_matrix_completion(M_path, Omega_M_path, C, ...
                                                lambda1, lambda2, T, ...
                                                eta_t, decay_rate, epsilon, ...
                                                clip_or_sqhinge, ...
                                                initialization, scale,stop_eps)
    %% DTr minimization based on subgradient descent
    % Min {f(X) + \lambda_1 \|X\|_* + \lambda_2 \|W(X)\hadamard(X - C)\|_*}
    %

    if(strcmp(clip_or_sqhinge, 'sqhinge'))
        update = @update_basic;
        get_obj_value = @get_value_basic;
    end

    C = double(C);
    eta_t = double(eta_t);
    eta_zero = eta_t;
    lambda1 = double(lambda1);
    lambda2 = double(lambda2);
    decay_rate = double(decay_rate);
    scale = double(scale);
    stop_eps = double(stop_eps);

    % M_path contains M
    load(M_path);
    load(Omega_M_path);

    [n, d] = size(M);
    rng(1);
    if (strcmp(initialization, 'zeros'))
        sol_M=zeros(n,d);
    elseif(strcmp(initialization, 'ones'))
        sol_M= ones(n,d);
    elseif(strcmp(initialization, 'large'))
        sol_M=(C + scale) * ones(n,d);
    end

    R_C = (M == C);

    [h_ret,h_U,h_S,h_V]=h_function(sol_M, C);

    old_X = 0;
    rel_scale = scale * numel(M);

    for i=1:T
        %% Update
        [sol_M,trnrm] = update(sol_M, C, M, Omega_M, R_C, h_U, h_S, h_V, lambda1, lambda2, eta_t, epsilon);

        %% Evaluate
        % Calculate current objective value
        if (mod(i,10)==0)
            disp(["Round: " num2str(i)]);
            [obj, f_value, trnrm_term, h_ret_term] = get_obj_value(sol_M, C, M, Omega_M, R_C, trnrm, lambda1, lambda2);

            fprintf('%f = %f + %f + %f', obj, f_value, trnrm_term, h_ret_term);
            fprintf('\n');
        end

        %% Update
        eta_t=eta_t*decay_rate;
    end

    disp(["Done"]);
end
function [sol_M,trnrm]=update_basic(sol_M, C, M, Omega_M, R_C, h_U, h_S, h_V, lambda1, ...
                                    lambda2, eta_t, epsilon)
    % Take gradient of f and h
    df = f_derivative_sqhinge(sol_M, C, M,Omega_M, R_C);
    dh = h_subgrad(sol_M, C,h_U,h_S,h_V);
    subgradient = df + lambda2 * dh;

    % Take gradient of \|X\|_*
    svd_obj = sol_M - eta_t * subgradient;
    [sol_M, trnrm] = shrink_trnrm(svd_obj, eta_t, lambda1, epsilon);
end
function [obj, f_value, trnrm_term, h_ret_term]=get_value_basic(sol_M, C, M, Omega_M, R_C, trnrm, lambda1, lambda2)
    [h_ret, h_U, h_S, h_V] = h_function(sol_M, C);
    f_value = f_function_sqhinge(sol_M, C, M,Omega_M, R_C);
    trnrm_term = lambda1 * trnrm;
    h_ret_term = lambda2 * h_ret;
    obj = f_value + trnrm_term + h_ret_term;
end