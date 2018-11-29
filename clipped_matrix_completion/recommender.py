from clipped_matrix_completion.util.cast import cast_to_rating
from sklearn.base import BaseEstimator


class Recommender(BaseEstimator):
    ## BEGIN for sklearn
    def __init__(self,
                 rank=None,
                 max_rank=None,
                 max_iter=None,
                 lam=None,
                 censoring_threshold=None,
                 test=None,
                 kappa=None,
                 delta_divisor=None,
                 sigma=None,
                 gpu=-1,
                 Wrow=None,
                 Wcol=None,
                 lambda1=None,
                 lambda2=None,
                 eta_t=None,
                 decay_rate=None,
                 loss_type=None,
                 dataname=None,
                 tol=None,
                 initialization=None,
                 niter=None,
                 verbose=False,
                 debug=False):
        self.rank = rank
        self.max_rank = max_rank
        self.max_iter = max_iter
        self.niter = niter
        self.lam = lam
        self.test = test
        self.censoring_threshold = censoring_threshold
        self.delta_divisor = delta_divisor
        self.kappa = kappa
        self.sigma = sigma
        self.gpu = gpu
        self.verbose = verbose
        self.debug = debug
        self.Wrow = Wrow
        self.Wcol = Wcol
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.loss_type = loss_type
        self.eta_t = eta_t
        self.decay_rate = decay_rate
        self.dataname = dataname
        self.tol = tol
        self.initialization = initialization

    def fit(self, R, R_again=None):
        R = R[0][0]
        self.P, self.Q = self.factorize(R)
        return self

    def predict(self, in_data):
        return self.predict_all()

    ## end for sklearn

    def factorize(self, R):
        raise NotImplementedError()

    def predict_all(self):
        raise NotImplementedError()

    def predict_original_all(self):
        return self.predict_all()

    def predict_truncated_all(self, rating_options):
        return cast_to_rating(self.predict_original_all(), rating_options)
