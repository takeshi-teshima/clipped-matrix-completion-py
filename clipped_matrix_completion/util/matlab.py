import os
import numpy as np
from scipy.io import savemat
from datetime import datetime
from scipy import sparse


def save_as_sparse_matrix(R, matrix_name, writable_mat_file_path):
    indices = np.nonzero(~np.isnan(R))
    R = sparse.coo_matrix(
        (R[indices], indices), shape=R.shape, dtype='double').tocsr()
    savemat(writable_mat_file_path, {matrix_name: R})
    return writable_mat_file_path
