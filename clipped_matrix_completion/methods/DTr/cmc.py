import os
import numpy as np
import matlab
import matlab.engine
from utillib.matlab import save_as_sparse_matrix
from config import TMP_DIR
dir_file_path = os.path.dirname(os.path.realpath(__file__))
import time


MATLAB_MATRIX_NAME = 'M'
MATLAB_MASK_NAME = 'Omega_M'


def clipping_aware_matrix_completion(R,
                                     clipping_threshold,
                                     lambda1=None,
                                     lambda2=None,
                                     T=None,
                                     eta_t=None,
                                     decay_rate=None,
                                     eps=None,
                                     loss_type=None,
                                     initialization=None,
                                     scale=None,
                                     stop_eps=None):
    if lambda1 is None:
        lambda1 = 0.01
    if lambda2 is None:
        lambda2 = 0.01
    if decay_rate is None:
        decay_rate = 0.99
    if eta_t is None:
        eta_t = 100
    if T is None:
        T = 500
    if loss_type is None:
        loss_type = 'sqhinge'
    if initialization is None:
        initialization = 'zeros'
    if eps is None:
        eps = 1e-8
    if scale is None:
        scale = 1.0
    if stop_eps is None:
        stop_eps = 1e-3
    eng = matlab.engine.start_matlab()
    eng.cd(dir_file_path + '/' + 'algorithm')

    M_path = save_as_sparse_matrix(R, MATLAB_MATRIX_NAME, TMP_DIR)
    Omega_M_path = save_as_sparse_matrix(~np.isnan(R), MATLAB_MASK_NAME,
                                         TMP_DIR)
    time.sleep(1)  # to avoid "*.mat not found" error
    M_hat = eng.clipping_aware_matrix_completion(
        M_path, Omega_M_path, clipping_threshold, lambda1, lambda2, T, eta_t,
        decay_rate, eps, loss_type, initialization, scale, stop_eps)
    os.remove(M_path)
    os.remove(Omega_M_path)

    return np.array(M_hat)
