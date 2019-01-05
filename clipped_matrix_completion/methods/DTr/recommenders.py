import numpy as np
from clipped_matrix_completion.recommender import Recommender
from .cmc import clipping_aware_matrix_completion


class DoubleTrCMFRecommender(Recommender):
    def factorize(self, R):
        R = R.copy()
        self.pred = clipping_aware_matrix_completion(
            R, self.censoring_threshold, self.lambda1, self.lambda2,
            self.lambda3, self.max_iter, self.eta_t, self.decay_rate,
            self.loss_type, self.initialization, self.initial_margin)
        self.P = 'Parametrization is X'
        self.Q = 'Parametrization is X'

        return self.P, self.Q

    def predict(self, in_data):
        return self.predict_all()

    def predict_all(self):
        return self.pred
