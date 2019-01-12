from .cmc import censoring_aware_low_rank_approx
from .cmc_without_jit import censoring_aware_low_rank_approx as censoring_aware_low_rank_approx_without_numba
from .cmc_sparse import cmc_sparse
import numpy as np
from clipped_matrix_completion.recommender import Recommender


class FroMCRecommender(Recommender):
    def factorize(self, R):
        R = R.copy()
        P, Q = censoring_aware_low_rank_approx(
            R,
            self.rank,
            censoring_threshold=None,
            init_size=self.censoring_threshold,
            lam=self.lam,
            max_iter=self.max_iter,
            # use_convergence_criteria=True,
            use_convergence_criteria=False,
            initU=self.initialization,
            initV=self.initialization,
            verbose=self.verbose)
        self.P, self.Q = P, Q.T
        return self.P, self.Q

    def predict_all(self):
        return self.P @ self.Q.T


class FroMCWithIgnoreRecommender(Recommender):
    def factorize(self, R):
        R = R.copy()
        R[R == self.censoring_threshold] = np.nan

        P, Q = censoring_aware_low_rank_approx(
            R,
            self.rank,
            censoring_threshold=None,
            init_size=self.censoring_threshold,
            lam=self.lam,
            initU=self.initialization,
            initV=self.initialization,
            max_iter=self.max_iter,
            # use_convergence_criteria=True,
            use_convergence_criteria=False,
            verbose=self.verbose)
        self.P, self.Q = P, Q.T
        return self.P, self.Q

    def predict_all(self):
        return self.P @ self.Q.T


class FroCMCRecommender(Recommender):
    def factorize(self, R):
        R = R.copy()
        P, Q = censoring_aware_low_rank_approx(
            R,
            self.rank,
            censoring_threshold=self.censoring_threshold,
            init_size=self.censoring_threshold,
            lam=self.lam,
            max_iter=self.max_iter,
            # initU='auto',
            # initV='auto',
            initU=self.initialization,
            initV=self.initialization,
            # use_convergence_criteria=True,
            use_convergence_criteria=False,
            debug=self.debug,
            verbose=self.verbose)
        self.P, self.Q = P, Q.T
        return self.P, self.Q

    def predict_all(self):
        return self.P @ self.Q.T


class SparseFroCMCRecommender(Recommender):
    def factorize(self, R):
        R = R.copy()
        P, Q = cmc_sparse(
            R,
            self.rank,
            censoring_threshold=self.censoring_threshold,
            init_size=self.censoring_threshold,
            lam=self.lam,
            max_iter=self.max_iter,
            # initU='auto',
            # initV='auto',
            initU=self.initialization,
            initV=self.initialization,
            # use_convergence_criteria=True,
            use_convergence_criteria=False,
            debug=self.debug,
            verbose=self.verbose)
        self.P, self.Q = P, Q.T
        return self.P, self.Q

    def predict_all(self):
        return self.P @ self.Q.T


class FroCMCRecommenderWithoutNumba(Recommender):
    def factorize(self, R):
        R = R.copy()
        P, Q = censoring_aware_low_rank_approx_without_numba(
            R,
            self.rank,
            censoring_threshold=self.censoring_threshold,
            init_size=self.censoring_threshold,
            lam=self.lam,
            max_iter=self.max_iter,
            # initU='auto',
            # initV='auto',
            initU=self.initialization,
            initV=self.initialization,
            # use_convergence_criteria=True,
            use_convergence_criteria=False,
            debug=self.debug,
            verbose=self.verbose)
        self.P, self.Q = P, Q.T
        return self.P, self.Q

    def predict_all(self):
        return self.P @ self.Q.T
