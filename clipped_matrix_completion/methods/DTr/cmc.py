import os
import numpy as np
import matlab
import matlab.engine
from utillib.matlab import save_as_sparse_matrix
from tempfile import NamedTemporaryFile
dir_file_path = os.path.dirname(os.path.realpath(__file__))
import time
# from config import TMP_DIR

MATLAB_MATRIX_NAME = 'M'
MATLAB_MASK_NAME = 'Omega_M'


def clipping_aware_matrix_completion(R,
                                     clipping_threshold,
                                     loss_scaling_factor=0.01,
                                     lambda1=0.01,
                                     lambda2=0.01,
                                     T=500,
                                     eta_t=100,
                                     decay_rate=0.99,
                                     loss_type='clip',
                                     initialization='zeros',
                                     initial_margin=1.):

    eng = matlab.engine.start_matlab()
    eng.cd(dir_file_path + '/' + 'algorithm')

    with NamedTemporaryFile(suffix='.mat') as M_mat:
        M_path = save_as_sparse_matrix(R, MATLAB_MATRIX_NAME, M_mat.name)
        with NamedTemporaryFile(suffix='.mat') as Omega_mat:
            Omega_M_path = save_as_sparse_matrix(
                ~np.isnan(R), MATLAB_MASK_NAME, Omega_mat.name)
            time.sleep(1)  # to avoid "*.mat not found" error
            M_hat = eng.clipping_aware_matrix_completion(
                M_path, Omega_M_path, clipping_threshold, loss_scaling_factor,
                lambda1, lambda2, T, eta_t, decay_rate, loss_type,
                initialization, initial_margin)

    return np.array(M_hat)
