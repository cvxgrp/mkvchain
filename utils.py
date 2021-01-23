import numpy as np
from itertools import product

def mmc(Ps, rev=False):
    n = Ps[0].shape[0]
    P = np.eye(n)
    yield P
    for Pi in Ps:
        if rev:
            P = P @ Pi
        else:
            P = Pi @ P
        yield P

def to_dataset_ignore_na(states, features, n):
    T = len(states)
    p = features.shape[1]

    l = []
    t = 0
    while t < T - 1:
        if ~np.isnan(states[t]) and ~np.isnan(states[t+1]):
            e = np.zeros(n)
            e[states[t+1]] = 1.
            l.append((features[t], 1.0, states[t], e))
        t += 1
    return l

def to_dataset(Ps, states, features):
    T = len(Ps) + 1
    n = Ps[0].shape[0]
    p = features.shape[1]

    l = []

    # first part
    first_non_nan = 0
    while np.isnan(states[first_non_nan]):
        first_non_nan += 1
    
    if first_non_nan > 0:
        Ps_cum = list(mmc(Ps[:first_non_nan]))
        Ps_cum_reverse = list(mmc(Ps[:first_non_nan][::-1], rev=True))

        P = np.outer(np.ones(n), Ps_cum_reverse[first_non_nan-1][states[first_non_nan]]) * Ps[0].T
        P /= P.sum()
        s_first = P.sum(axis=1)
        s_last = np.zeros(n)
        s_last[states[first_non_nan]] = 1
        l = []
        Ttemp = first_non_nan + 1
        for t in range(Ttemp-1):
            P1 = Ps_cum_reverse[Ttemp-t-2]
            P2 = Ps_cum[t]
            Ptemps = [np.outer(P2[:,i], s_last @ P1) * Ps[t].T for i in range(n)]
            Ptemps = [P / P.sum() for P in Ptemps]
            P = sum([P * p for P, p in zip(Ptemps, s_first)])
            for r in range(n):
                if P[r].sum() > 0:
                    l.append((features[t], P[r].sum(), r, P[r] / P[r].sum()))

    # middle part
    last_non_nan = T-1
    while np.isnan(states[last_non_nan]):
        last_non_nan -= 1

    i = first_non_nan+0
    while i < last_non_nan:
        if ~np.isnan(states[i+1]):
            e = np.zeros(n)
            e[states[i+1]] = 1.
            l.append((features[i], 1.0, states[i], e))
            i += 1
        else:
            j = i+1
            while np.isnan(states[j]):
                j += 1
            s_first = np.zeros(n)
            s_first[states[i]] = 1.
            s_last = np.zeros(n)
            s_last[states[j]] = 1.
            Ps_cum = list(mmc(Ps[i:j]))
            Ps_cum_reverse = list(mmc(Ps[i:j][::-1], rev=True))
            Ttemp = j - i + 1
            for t in range(Ttemp-1):
                P1 = Ps_cum_reverse[Ttemp-t-2]
                P2 = Ps_cum[t]
                P = np.outer(P2 @ s_first, s_last @ P1) * Ps[i+t].T
                P /= P.sum()
                for r in range(n):
                    if P[r].sum() > 0:
                        l.append((features[i+t],P[r].sum(), r, P[r] / P[r].sum()))
            i = j

    # end part
    for t in range(last_non_nan, T-1):
        if t == last_non_nan:
            z = np.zeros((n, 1))
            z[states[last_non_nan]] = 1
        else:
            z = np.linalg.multi_dot([np.eye(n)] + Ps[last_non_nan:t][::-1])[:,states[last_non_nan]][:,None]
        P = z * Ps[t].T
        for r in range(n):
            if P[r].sum() > 0:
                l.append((features[t], P[r].sum(), r, P[r] / P[r].sum()))
    
    return l

def to_dataset_brute_force(Ps, states, features):
    T = len(Ps) + 1
    n = Ps[0].shape[0]
    p = features.shape[1]

    d = {}
    for seq in product(range(n), repeat=T):
        skip = False
        for t in range(T):
            if ~np.isnan(states[t]) and states[t] != seq[t]:
                skip = True
        if skip:
            continue
        prob = 1.
        for t in range(T-1):
            prob *= Ps[t][seq[t+1], seq[t]]
        d[seq] = prob

    l = []
    for t in range(T-1):
        P = np.zeros((n, n))
        for pair in product(range(n), repeat=2):
            for seq in d.keys():
                if seq[t] == pair[0] and seq[t+1] == pair[1]:
                    P[pair] += d[seq]
        P /= P.sum()
        for r in range(n):
            if P[r].sum() > 0:
                l.append((features[t], P[r].sum(), r, P[r] / P[r].sum()))

    return l

if __name__ == "__main__":
    np.random.seed(2)
    T = 7
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

        for i in np.random.choice(np.arange(T), np.random.randint(0, T)):
            states[i] = np.nan
        l1 = to_dataset_brute_force(Ps, states, features)
        l2 = to_dataset(Ps, states, features)
        for x, y in zip(l1, l2):
            for a, b in zip(x, y):
                np.testing.assert_allclose(a, b)