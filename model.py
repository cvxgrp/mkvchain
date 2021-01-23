import torch
import numpy as np
import warnings
from scipy.special import softmax

from utils import to_dataset, to_dataset_ignore_na

class FeatureDependentMarkovChain():
    def __init__(self, num_states, n_iter=50, lam=0.1, eps=1e-6):
        """
        Args:
            - num_states
            - n_iter
            - lam
            - eps
        """
        self.n = num_states
        self.n_iter = n_iter
        self.lam = lam
        self.eps = eps

    def fit(self, states, features, lengths, verbose=False):
        """
        Args:
            - states: numpy array of states
            - features: numpy array of features
            - lengths: numpy array of lengths of each sequence
        """
        N, p = features.shape

        self.models = {}
        prev_loss = float("inf")
        for k in range(self.n_iter):
            X = dict([(i, []) for i in range(self.n)])
            Y = dict([(i, []) for i in range(self.n)])
            weights = dict([(i, []) for i in range(self.n)])
            i = 0
            for length in lengths:
                s = states[i:i+length]
                f = features[i:i+length]
                if k == 0:
                    l = to_dataset_ignore_na(s, f, self.n)
                else:
                    # Get Ps
                    Ps = self.predict(f[:-1])
                    l = to_dataset(list(Ps), s, f)

                for feat, w, state, next_state in l:
                    X[state].append(feat)
                    Y[state].append(next_state)
                    weights[state].append(w)
                i += length

            loss = 0.
            for i in range(self.n):
                if len(weights[i]) == 0: # no data points
                    warnings.warn("No pairs found in the dataset; results might be innacurate or useless.")
                    A = np.zeros((p, self.n))
                    b = np.zeros(self.n)
                    l = 0.
                else:
                    weightsi, Xi, Yi = np.array(weights[i]), np.array(X[i]), np.array(Y[i])
                    A, b, l = self._logistic_regression(weightsi, Xi, Yi, self.lam)
                self.models[i] = (A, b)
                loss += l

            if k > 0 and loss <= prev_loss and 1 - loss / prev_loss <= self.eps:
                break
            if k > 0:
                if verbose:
                    print(k, loss)
            if k > 0:
                prev_loss = loss

    def _logistic_regression(self, weights, X, Y, lam):
        torch.set_default_dtype(torch.double)
        weights = torch.from_numpy(weights)
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        N = X.shape[0]

        A = torch.zeros(X.shape[1], Y.shape[1], requires_grad=True)
        b = torch.zeros(Y.shape[1], requires_grad=True)
        opt = torch.optim.LBFGS([A, b], line_search_fn='strong_wolfe')
        loss_fn = torch.nn.KLDivLoss(reduction='none')
        lsm = torch.nn.LogSoftmax(dim=1)

        def loss():
            opt.zero_grad()
            pred = lsm(X @ A + b)
            l = loss_fn(pred, Y).sum(axis=1)
            l = (l * weights).sum() + (lam / 2) * A.pow(2).sum()
            l.backward()
            return l

        opt.step(loss)

        A_numpy = A.detach().numpy()    
        b_numpy = b.detach().numpy()
        return (A_numpy, b_numpy, loss().item())

    def predict(self, features):
        return np.array([softmax(features @ self.models[i][0] + self.models[i][1], axis=1) for i in range(self.n)]).swapaxes(0,1).swapaxes(1,2)

    def score(self, states, features, lengths):
        X = dict([(i, []) for i in range(self.n)])
        Y = dict([(i, []) for i in range(self.n)])
        i = 0
        for length in lengths:
            s = states[i:i+length]
            f = features[i:i+length]
            l = to_dataset_ignore_na(s, f, self.n)

            for feat, w, state, next_state in l:
                X[state].append(feat)
                Y[state].append(next_state)
            i += length

        ll = 0.
        for i in range(self.n):
            if len(X[i]) == 0:
                continue
            Ps = self.predict(np.array(X[i]))
            ll += (np.log(Ps[:, :, i]) * np.array(Y[i])).sum() / len(X[i])

        return ll



if __name__ == "__main__":
    np.random.seed(2)
    T = 40
    n = 2
    features = np.random.randn(T, 3)

    for _ in range(100):
        Ps = []
        for t in range(T-1):
            P = np.random.rand(n, n)
            P /= P.sum(axis=0)
            Ps.append(P)
        s = 0
        states = [s]
        for t in range(T-1):
            s = np.random.choice(np.arange(n), p=Ps[t][:,s])
            states.append(s)

        for i in np.random.choice(np.arange(1, T-1), np.random.randint(0, T)):
            states[i] = np.nan

    model = FeatureDependentMarkovChain(n, 50)
    model.fit(states, features, [len(states)], verbose=True)