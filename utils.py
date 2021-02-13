import numpy as np
from itertools import product

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

def to_dataset(Ps, states, features):
    T = len(Ps) + 1
    n = Ps[0].shape[0]
    p = features.shape[1]
    assert len(states) == T
    assert features.shape[0] == T
    assert not np.isnan(states[0]) and not np.isnan(states[-1])

    Ps = [P.T for P in Ps]

    l = []

    i = 0
    while i < T - 1:
        assert not np.isnan(states[i])
        # deterministic transition
        if ~np.isnan(states[i+1]):
            e = np.zeros(n)
            e[states[i+1]] = 1.
            l.append((features[i], 1.0, states[i], e))
            i += 1
        # sequence of unknown transitions
        else:
            # find next known
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
    
    return l

def to_dataset_brute_force(Ps, states, features):
    T = len(Ps) + 1
    n = Ps[0].shape[0]
    p = features.shape[1]
    assert len(states) == T
    assert features.shape[0] == T

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
            prob *= Ps[t][seq[t], seq[t+1]]
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
    T = 8
    n = 2
    features = np.random.randn(T, 3)

    for _ in range(100):
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
        l1 = to_dataset_brute_force(Ps, states, features)
        l2 = to_dataset(Ps, states, features)
        for x, y in zip(l1, l2):
            for a, b in zip(x, y):
                np.testing.assert_allclose(a, b)