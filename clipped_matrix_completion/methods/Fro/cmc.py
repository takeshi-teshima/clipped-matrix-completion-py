import numpy as np
from numpy.linalg import norm
import warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.sparse.linalg import svds
from numba import jit
from clipped_matrix_completion.util.suppressor import Suppressor

@jit(
    'f8[:, :](f8[:, :], f8[:, :], f8[:, :], b1[:, :], b1[:, :], f8[:,:], i8)',
    nopython=True,
    cache=True,
    # parallel=True
)
def update_u_heuristic(U, V, mat, obs_not_censored, obs_and_censored, lamE,
                       n_1):
    for i in range(n_1):
        v_obs = V[:, obs_not_censored[i]]
        v_cens = V[:, obs_and_censored[i]]
        mat_obs = mat[i][obs_not_censored[i]]
        mat_cens = mat[i][obs_and_censored[i]]
        W = (mat_cens >= U[i] @ v_cens)

        v_a = v_obs @ v_obs.T
        if v_obs.size == 0:
            v_a = np.zeros(v_a.shape, np.float64)

        vc_a = v_cens[:, W] @ v_cens[:, W].T
        if v_cens[:, W].size == 0:
            vc_a = np.zeros(vc_a.shape, np.float64)

        prod_obs = mat_obs @ v_obs.T
        if v_obs.size == 0:
            prod_obs = np.zeros(prod_obs.shape, dtype=np.float64)

        prod_cens = mat_cens[W] @ v_cens[:, W].T
        if v_cens[:, W].size == 0:
            prod_cens = np.zeros(prod_cens.shape, dtype=np.float64)

        U[i] = np.linalg.solve(v_a + vc_a + lamE, prod_obs + prod_cens)
    return U


@jit(
    'f8[:, :](f8[:, :], f8[:, :], f8[:, :], b1[:, :], b1[:, :], f8[:,:], i8)',
    nopython=True,
    cache=True,
    # parallel=True
)
def update_v_heuristic(U, V, mat, obs_not_censored, obs_and_censored, lamE,
                       n_2):
    for j in range(n_2):
        u_obs = U[obs_not_censored[:, j]]
        u_cens = U[obs_and_censored[:, j]]
        mat_obs = mat[obs_not_censored[:, j]][:, j]
        mat_cens = mat[obs_and_censored[:, j]][:, j]
        W = (mat_cens >= u_cens @ V[:, j])

        u_a = u_obs.T @ u_obs
        if u_obs.size == 0:
            u_a = np.zeros(u_a.shape, dtype=np.float64)

        uc_a = u_cens[W].T @ u_cens[W]
        if u_cens[W].size == 0:
            uc_a = np.zeros(uc_a.shape, dtype=np.float64)

        prod_obs = u_obs.T @ mat_obs
        if u_obs.size == 0:
            prod_obs = np.zeros(prod_obs.shape, dtype=np.float64)

        prod_cens = u_cens[W].T @ mat_cens[W]
        if u_cens[W].size == 0:
            prod_cens = np.zeros(prod_cens.shape, dtype=np.float64)

        V[:, j] = np.linalg.solve(u_a + uc_a + lamE, prod_obs + prod_cens)
    return V


@jit
def auto_init_v(mat, observed, U, lam, rank, n_2):
    V = np.zeros((rank, n_2))
    for j in range(n_2):
        u_obs = U[observed[:, j]]
        mat_obs = mat[observed[:, j], j]
        V[:, j] = np.linalg.solve((u_obs.T @ u_obs + lam * np.identity(rank)),
                                  u_obs.T @ mat_obs)
    return V


def censoring_aware_low_rank_approx(mat,
                                    rank,
                                    initU=None,
                                    initV=None,
                                    censoring_threshold=5,
                                    init_size=5,
                                    init_scale=1.0,
                                    lam=0.1,
                                    epsilon=1e-3,
                                    max_iter=200,
                                    use_convergence_criteria=True,
                                    convergence_check_frequency=100,
                                    verbose=False,
                                    debug=False):
    """
    mat: np.nan if missing
    initU \in \mathbb{R}^{n_1 \times r}
    initV \in \mathbb{R}^{r \times n_2}
    """
    n_1, n_2 = mat.shape

    observed = ~np.isnan(mat)
    censored = (mat == censoring_threshold)
    obs_not_censored = observed * ~censored
    obs_and_censored = observed * censored

    if (initU is None) or (initU == 'auto'):
        # this rank should be the true rank?
        U, _, _ = svds(np.nan_to_num(mat), rank)
    elif initU == 'unit':
        U = np.ones((mat.shape[0], rank)) * 1. / rank
    elif initU == 'proposed':
        U = np.ones((mat.shape[0], rank)) * np.sqrt(
            (init_size + init_scale) / rank)
    elif initU == 'random':
        U = np.random.normal(scale=1. / rank, size=(mat.shape[0], rank))
    else:
        U = initU.copy()

    if (initV is None) or (initV == 'auto'):
        V = auto_init_v(mat, observed, U, lam, rank, mat.shape[1])
    elif initV == 'unit':
        V = np.ones((rank, mat.shape[1])) * 1. / rank
    elif initV == 'proposed':
        V = np.ones((rank, mat.shape[1])) * np.sqrt(
            (init_size + init_scale) / rank)
    elif initV == 'random':
        V = np.random.normal(scale=1. / rank, size=(rank, mat.shape[1]))
    else:
        V = initV.copy()

    ################################
    # For acceleration using Numba
    mat[np.isnan(mat)] = 0.
    mat = mat.astype(np.float64)
    U = U.astype(np.float64)
    V = V.astype(np.float64)
    lam = np.float64(lam)
    n_1, n_2 = np.int64(n_1), np.int64(n_2)
    lamE = (lam * np.identity(np.int64(rank))).astype(np.float64)

    U_shape = U.shape
    V_shape = V.shape
    # For suppressing MKL errors
    suppressor = Suppressor()
    ################################

    # norm_mat_obs = np.sum(mat[observed]**2)
    norm_mat_obs = np.sum(observed)

    loss1 = lambda U, V: np.sum((mat[obs_not_censored] - (U @ V)[obs_not_censored])**2)
    loss2 = lambda U, V: np.sum((np.maximum(0, mat[obs_and_censored] - (U @ V)[obs_and_censored]))**2)
    reg = lambda U, V: lam * (np.sum(U**2) + np.sum(V**2))
    if censoring_threshold is not None:
        loss = lambda U, V: loss1(U, V) + loss2(U, V) + reg(U, V)
        main_loss = lambda U, V: loss1(U, V) + loss2(U, V)
    else:
        loss = lambda U, V: loss1(U, V) + reg(U, V)
        main_loss = lambda U, V: loss1(U, V)

    if use_convergence_criteria:
        prev_criteria = loss(U, V) / norm_mat_obs

    for iteration in range(1, max_iter + 1):
        suppressor.suppress()
        # fix U, update V
        V = update_v_heuristic(U, V, mat, obs_not_censored, obs_and_censored,
                               lamE, n_2)

        # fix V, update U
        U = update_u_heuristic(U, V, mat, obs_not_censored, obs_and_censored,
                               lamE, n_1)

        if (use_convergence_criteria) and (
                iteration % convergence_check_frequency == 0):
            current_criteria = loss(U, V) / norm_mat_obs
        suppressor.unsuppress()
        if (use_convergence_criteria) and (
                iteration % convergence_check_frequency == 0):
            if (prev_criteria - current_criteria <= epsilon):
                print('converged.')
                break
            prev_criteria = current_criteria.copy()

        if iteration == max_iter and epsilon > 0:
            warnings.warn(
                "Maximum number of iteration %d reached. Increase it to"
                " improve convergence." % max_iter, ConvergenceWarning)
        if verbose:
            if (verbose == 'more') or (iteration % 10 == 0):
                err = loss(U, V)
                sq = loss1(U, V)
                hinge = loss2(U, V)
                regul = reg(U, V)
                print('iter: ', iteration, ' normalized loss: ',
                      err / norm_mat_obs, 'sq loss', sq, 'hinge', hinge,
                      'reg / loss', regul / err)
    suppressor.unsuppress()
    suppressor.close()
    return U, V
