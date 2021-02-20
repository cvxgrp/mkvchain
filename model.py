import torch
import numpy as np
import warnings
from scipy.special import softmax
from optimizer import AdaptiveProximalGradient as APG

from utils import to_dataset, to_dataset_ignore_na

class FeatureDependentMarkovChain():
    def __init__(self, num_states, mask=None, lam_frob=0.1, W_lap_states=None,
                 W_lap_features=None, lam_col_norm=0.0, eps=1e-6, n_iter=50):
        """
        Args:
            - num_states
            - mask
            - lam_frob
            - W_lap_states
            - W_lap_features
            - eps
            - n_iter
        """
        self.n = num_states
        self.n_iter = n_iter
        self.lam = lam_frob
        self.W_lap_states = W_lap_states
        self.W_lap_features = W_lap_features
        self.lam_col_norm = lam_col_norm
        self.eps = eps
        if mask is None:
            self.mask = np.ones((self.n, self.n))
        else:
            assert mask.shape == (self.n, self.n)
            self.mask = mask
        self.nonzero = [np.where(self.mask[i])[0] for i in range(self.n)]
        self.zero = [np.where(1-self.mask[i])[0] for i in range(self.n)]
        self.sizes = [len(n) for n in self.nonzero]
        for s in self.sizes:
            assert s > 0, "Mask must have at least one state > 0"

    def fit(self, states, features, lengths, verbose=False, warm_start=False, **kwargs):
        """
        Args:
            - states: numpy array of states
            - features: numpy array of features
            - lengths: numpy array of lengths of each sequence
            - verbose:
            - warm_start: 
        """
        N, m = features.shape

        self.models = {}
        prev_loss = float("inf")
        for k in range(self.n_iter):
            X = dict([(i, []) for i in range(self.n)])
            Y = dict([(i, []) for i in range(self.n)])
            weights = dict([(i, []) for i in range(self.n)])
            i = 0
            for length in lengths:
                if length <= 1:
                    i += length
                    continue
                s = states[i:i+length]
                f = features[i:i+length]
                if k == 0:
                    l = to_dataset_ignore_na(s, f, self.n)
                else:
                    # Get Ps
                    Ps = self.predict(f[:-1])
                    l = to_dataset(list(Ps), s, f)

                for feat, w, state, next_state in l:
                    zero = self.zero[state]
                    if np.any(next_state[zero] > 0):
                        warnings.warn("Transition from " + str(state) + " to " + str(next_state) + " impossible according to mask. Ignoring transition.")
                        continue
                    if np.any(np.isnan(feat)):
                        continue
                    X[state].append(feat)
                    Y[state].append(next_state)
                    weights[state].append(w)
                i += length

            ws, Xs, Ys = [], [], []
            for i in range(self.n):
                noutputs = self.sizes[i]
                if len(weights[i]) == 0: # no data points
                    warnings.warn("No pairs found in the dataset starting from state " + \
                        str(i) + " . results starting from this state might be innacurate or useless.")
                    weightsi = np.ones(1)
                    Xi = np.zeros((1, m))
                    Yi = np.zeros((1, noutputs))
                    Yi[self.nonzeros[i]] = 1 / len(self.nonzeros[i])
                else:
                    weightsi, Xi, Yi = np.array(weights[i]), np.array(X[i]), np.array(Y[i])
                ws.append(weightsi)
                Xs.append(Xi)
                Ys.append(Yi[:, self.nonzero[i]])

            if self.lam_col_norm == 0.0:
                self.As, self.bs, loss = self._logistic_regression(ws, Xs, Ys, self.lam, warm_start=warm_start,
                                                   W_lap_states=self.W_lap_states, W_lap_features=self.W_lap_features, **kwargs)
            else:
                self.As, self.bs, loss = self._logistic_regression_column_norm(ws, Xs, Ys, self.lam, warm_start=warm_start,
                                                   W_lap_states=self.W_lap_states, W_lap_features=self.W_lap_features,
                                                   lam_col_norm=self.lam_col_norm, **kwargs)
            if k > 0:
                if verbose:
                    print("%03d | %8.4e" % (k, -loss))
            if k > 0 and loss <= prev_loss and 1 - loss / prev_loss <= self.eps:
                break
            if k > 0:
                prev_loss = loss


    def _logistic_regression(self, ws, Xs, Ys, lam, warm_start=False,
                             W_lap_states=None, W_lap_features=None, **kwargs):
        torch.set_default_dtype(torch.double)
        ws = [torch.from_numpy(w) for w in ws]
        Xs = [torch.from_numpy(X) for X in Xs]
        Ys = [torch.from_numpy(Y) for Y in Ys]

        m = Xs[0].shape[1]

        if warm_start:
            assert hasattr(self, "As")
            assert hasattr(self, "bs")
            As = [torch.from_numpy(A) for A in self.As]
            bs = [torch.from_numpy(b) for b in self.bs]
            for A, b in zip(As, bs):
                A.requires_grad_(True)
                b.requires_grad_(True)
        else:
            As = [torch.zeros(m, Ys[i].shape[1], requires_grad=True) for i in range(self.n)]
            bs = [torch.zeros(Ys[i].shape[1], requires_grad=True) for i in range(self.n)]

        opt = torch.optim.LBFGS(As + bs, max_iter=250, tolerance_grad=1e-8, line_search_fn='strong_wolfe')
        # opt = torch.optim.SGD(As + bs, lr=1., momentum=.9)
        loss_fn = torch.nn.KLDivLoss(reduction='none')
        lsm = torch.nn.LogSoftmax(dim=1)

        divisor = sum([w.sum().item() for w in ws])

        def loss():
            opt.zero_grad()
            l = 0
            A = torch.zeros((self.n, m, self.n))
            b = torch.zeros((self.n, self.n))
            for i in range(self.n):
                A[i,:,self.nonzero[i]] += As[i]
                b[i,self.nonzero[i]] += bs[i]
                pred = lsm(Xs[i] @ As[i] + bs[i])
                l += (loss_fn(pred, Ys[i]).sum(axis=1) * ws[i]).sum() / divisor
                l += lam * As[i].pow(2).sum()
            if W_lap_states is not None:
                rows, cols = W_lap_states.nonzero()
                A_diff = (A[rows, :, :] - A[cols, :, :]).pow(2).sum((1, 2))
                b_diff = (b[rows] - b[cols]).pow(2).sum(1)
                l += ((A_diff + b_diff) * torch.from_numpy(W_lap_states.data)).sum()
            if W_lap_features is not None:
                rows, cols = W_lap_features.nonzero()
                A_diff = (A[:, rows, :] - A[:, cols, :]).pow(2).sum((0, 2))
                l += (A_diff * torch.from_numpy(W_lap_features.data)).sum()
            l.backward()
            # print(sum([A.grad.norm(2).sum().item() for A in As]), sum([b.grad.norm(2).sum().item() for b in bs]))
            return l

        opt.step(loss)

        A_numpy = [A.detach().numpy() for A in As]
        b_numpy = [b.detach().numpy() for b in bs]
        return (A_numpy, b_numpy, loss().item())

    def _logistic_regression_column_norm(self, ws, Xs, Ys, lam, warm_start=False,
                             W_lap_states=None, W_lap_features=None, lam_col_norm=.1):
        torch.set_default_dtype(torch.double)
        ws = [torch.from_numpy(w) for w in ws]
        Xs = [torch.from_numpy(X) for X in Xs]
        Ys = [torch.from_numpy(Y) for Y in Ys]

        m = Xs[0].shape[1]

        if warm_start:
            assert hasattr(self, "As")
            assert hasattr(self, "bs")
            As = [torch.from_numpy(A) for A in self.As]
            bs = [torch.from_numpy(b) for b in self.bs]
            for A, b in zip(As, bs):
                A.requires_grad_(True)
                b.requires_grad_(True)
        else:
            As = [torch.zeros(m, Ys[i].shape[1], requires_grad=True) for i in range(self.n)]
            bs = [torch.zeros(Ys[i].shape[1], requires_grad=True) for i in range(self.n)]

        loss_fn = torch.nn.KLDivLoss(reduction='none')
        lsm = torch.nn.LogSoftmax(dim=1)

        divisor = sum([w.sum().item() for w in ws])

        def loss():
            l = 0
            A = torch.zeros((self.n, m, self.n))
            b = torch.zeros((self.n, self.n))
            for i in range(self.n):
                A[i,:,self.nonzero[i]] += As[i]
                b[i,self.nonzero[i]] += bs[i]
                pred = lsm(Xs[i] @ As[i] + bs[i])
                l += (loss_fn(pred, Ys[i]).sum(axis=1) * ws[i]).sum() / divisor
                l += lam * As[i].pow(2).sum()
            if W_lap_states is not None:
                rows, cols = W_lap_states.nonzero()
                A_diff = (A[rows, :, :] - A[cols, :, :]).pow(2).sum((1, 2))
                b_diff = (b[rows] - b[cols]).pow(2).sum(1)
                l += ((A_diff + b_diff) * torch.from_numpy(W_lap_states.data)).sum()
            if W_lap_features is not None:
                rows, cols = W_lap_features.nonzero()
                A_diff = (A[:, rows, :] - A[:, cols, :]).pow(2).sum((0, 2))
                l += (A_diff * torch.from_numpy(W_lap_features.data)).sum()
            return l

        def r():
            r = 0.
            for i in range(m):
                r += torch.cat([A[i,:] for A in As]).norm()
            return lam_col_norm * r

        def prox(t):
            for i in range(m):
                r = torch.cat([A[i,:] for A in As]).norm()
                v_norm = r.item()
                if v_norm < lam_col_norm * t:
                    mult = 0
                else:
                    mult = 1 - lam_col_norm * t / v_norm
                for A in As:
                    A.data[i,:] *= mult

        t = 1.
        for k in range(40):
            for A, b in zip(As, bs):
                A.grad = None
                b.grad = None
            l = loss()
            l.backward()

            prev_As = [A.data.clone() for A in As]
            prev_bs = [b.data.clone() for b in bs]
            prev_loss = l.item() + r().item()
            print(k, t, prev_loss)
            while True:
                for i in range(self.n):
                    As[i].data = prev_As[i].data - t * As[i].grad
                    bs[i].data = prev_bs[i].data - t * bs[i].grad
                prox(t)
                cur_loss = loss().item() + r().item()
                if cur_loss <= prev_loss:
                    t *= 1.2
                    break
                else:
                    t /= 2
                if t <= 1e-8:
                    for i in range(self.n):
                        As[i].data = prev_As[i].data
                        bs[i].data = prev_bs[i].data
                    break

        A_numpy = [A.detach().numpy() for A in As]
        b_numpy = [b.detach().numpy() for b in bs]
        return (A_numpy, b_numpy, (loss() + r()).item())

    def predict(self, features):
        P = []
        for i in range(self.n):
            yi = np.zeros((features.shape[0], self.n))
            yi[:, self.nonzero[i]] = softmax(features @ self.As[i] + self.bs[i], axis=1)
            P.append(yi)
        P = np.array(P).swapaxes(0, 1)
        return P

    def score(self, states, features, lengths, average=False):
        X = dict([(i, []) for i in range(self.n)])
        Y = dict([(i, []) for i in range(self.n)])
        i = 0
        for length in lengths:
            if length <= 1:
                i += length
                continue
            s = states[i:i+length]
            f = features[i:i+length]
            l = to_dataset_ignore_na(s, f, self.n)

            for feat, w, state, next_state in l:
                if np.any(next_state[self.zero[state]] > 0):
                    warnings.warn("Transition from " + str(state) + " to " + str(next_state) + " impossible according to mask. Ignoring transition.")
                    continue
                if np.any(np.isnan(feat)):
                    continue
                X[state].append(feat)
                Y[state].append(next_state)
            i += length

        ll = 0.
        ct = 0
        for i in range(self.n):
            if len(X[i]) == 0:
                continue
            ct += len(X[i])
            Ps = self.predict(np.array(X[i]))
            z = np.log(Ps[:, i, :])
            z[z == -np.inf] = 0.
            ll += (z * np.array(Y[i])).sum()
        if average:
            ll /= ct
        return ll


if __name__ == "__main__":
    np.random.seed(2)
    T = 40
    n = 2
    features = np.random.randn(T, 3)

    Ps = []
    for t in range(T-1):
        P = np.random.rand(n, n)
        P /= P.sum(axis=1)[:, None]
        Ps.append(P)
    s = 0
    states = [s]
    for t in range(T-1):
        s = np.random.choice(np.arange(n), p=Ps[t][s,:])
        states.append(s)

    for i in np.random.choice(np.arange(1, T-1), np.random.randint(0, T)):
        states[i] = np.nan

    model = FeatureDependentMarkovChain(n, 50, eps=1e-6)
    model.fit(states, features, [len(states)], verbose=True)