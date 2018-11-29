import numpy as np
from clipped_matrix_completion.recommender import Recommender
from .cmc import clipping_aware_matrix_completion


class DoubleTrMFRecommender(Recommender):
    def factorize(self, R):
        raise NotImplementedError()
        R = R.copy()
        self.pred = matrix_completion(R)

        self.P = 'Parametrization is X'
        self.Q = 'Parametrization is X'

        return self.P, self.Q

    def predict(self, in_data):
        return self.predict_all()

    def predict_all(self):
        return self.pred


class DoubleTrMFWithIgnoreRecommender(Recommender):
    def factorize(self, R):
        raise NotImplementedError()
        R = R.copy()
        R[R == self.censoring_threshold] = np.nan

        self.pred = matrix_completion(R)

        self.P = 'Parametrization is X'
        self.Q = 'Parametrization is X'

        return self.P, self.Q

    def predict(self, in_data):
        return self.predict_all()

    def predict_all(self):
        return self.pred


class DoubleTrCMFRecommender(Recommender):
    def factorize(self, R):
        R = R.copy()
        self.pred = clipping_aware_matrix_completion(
            R,
            self.censoring_threshold,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            T=self.max_iter,
            eta_t=self.eta_t,
            decay_rate=self.decay_rate,
            loss_type=self.loss_type,
            initialization=self.initialization)
        self.P = 'Parametrization is X'
        self.Q = 'Parametrization is X'

        return self.P, self.Q

    def predict(self, in_data):
        return self.predict_all()

    def predict_all(self):
        return self.pred
