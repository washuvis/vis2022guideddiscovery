import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss
from scipy.optimize import minimize, Bounds

from tqdm import tqdm


class KNNModel:
    def __init__(self, alpha, weights):
        self.alpha = alpha
        self.weights = weights

    def predict(self, test_ind, train_ind, observed_labels):
        probs = np.empty((test_ind.size, 2))

        pos_ind = (observed_labels == 1)
        masks = [~pos_ind, pos_ind]

        csc_weights = self.weights[test_ind].tocsc()

        for class_ in range(2):
            probs[:, class_] = self.alpha[class_] + (
                csc_weights[:, train_ind[masks[class_]]]
                .sum(axis=1).flatten()
            )

        return normalize(probs, axis=1, norm='l1')[:, 1]

    def train(self, train_ind, observed_labels, verbose=False):
        return


class WeightedKNNModel:
    def __init__(self, alpha, text_weights, loc_weights, q=None, num_restarts=50):
        self.alpha = alpha

        self.text_weights = text_weights
        self.loc_weights = loc_weights

        if q is None:
            self.q = 0.5
        else:
            self.q = q

        self.num_restarts = num_restarts  # for optimizing q
        self.bounds = Bounds(0, 1)        # for optimizing q

    def old_predict(self, test_ind, train_ind, observed_labels):
        text_probs = np.empty((test_ind.size, 2))
        loc_probs = np.empty((test_ind.size, 2))

        pos_ind = (observed_labels == 1)
        masks = [~pos_ind, pos_ind]

        text_csc_weights = self.text_weights[test_ind].tocsc()
        loc_csc_weights = self.loc_weights[test_ind].tocsc()

        for class_ in range(2):
            tmp_train_ind = train_ind[masks[class_]]

            text_probs[:, class_] = self.alpha[class_] + (
                text_csc_weights[:, tmp_train_ind].sum(axis=1).flatten())

            loc_probs[:, class_] = self.alpha[class_] + (
                loc_csc_weights[:, tmp_train_ind].sum(axis=1).flatten())

        text_probs = normalize(text_probs, axis=1, norm='l1')[:, 1]
        loc_probs = normalize(loc_probs, axis=1, norm='l1')[:, 1]

        return self.q * text_probs + (1 - self.q) * loc_probs

    def predict(self, test_ind, train_ind, observed_labels):
        text_probs = np.empty((test_ind.size, 2))
        loc_probs = np.empty((test_ind.size, 2))

        pos_ind = (observed_labels == 1)
        masks = [~pos_ind, pos_ind]

        for class_ in range(2):
            tmp_train_ind = train_ind[masks[class_]]

            text_probs[:, class_] = self.alpha[class_] + (
                self.text_weights[:, tmp_train_ind][test_ind]
                .sum(axis=1).flatten()
            )

            loc_probs[:, class_] = self.alpha[class_] + (
                self.loc_weights[:, tmp_train_ind][test_ind]
                .sum(axis=1).flatten()
            )

        text_probs = normalize(text_probs, axis=1, norm='l1')[:, 1]
        loc_probs = normalize(loc_probs, axis=1, norm='l1')[:, 1]

        return self.q * text_probs + (1 - self.q) * loc_probs

    def old_train(self, train_ind, observed_labels, verbose=False):
        qs = np.linspace(0, 1, 101)[::-1]
        losses = np.empty(101)

        old_q = self.q

        for i, q in enumerate(qs):
            self.q = q
            probs = self.predict(train_ind, train_ind, observed_labels)
            losses[i] = log_loss(observed_labels, probs, labels=[0, 1])

        self.q = qs[np.argmin(losses)]

    def train(self, train_ind, observed_labels, verbose=False):
        def get_loss(q):
            old_q = self.q
            self.q = q

            probs = self.predict(train_ind, train_ind, observed_labels)

            self.q = old_q
            return log_loss(observed_labels, probs, labels=[0, 1])

        min_val = float('inf')
        best_q = None

        for q0 in np.random.uniform(size=self.num_restarts):
            res = minimize(
                get_loss, x0=[q0], bounds=self.bounds, method='L-BFGS-B')

            if res.fun < min_val:
                min_val = res.fun
                best_q = float(res.x)

        self.q = best_q
