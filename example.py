from model import FeatureDependentMarkovChain
import numpy as np
from scipy.special import softmax

np.set_printoptions(precision=3, suppress=True)

if __name__ == "__main__":
    np.random.seed(1)

    T = 200
    n = 3
    P = np.random.rand(n, n) + .1
    P /= P.sum(axis=0)

    s = 0
    states = [s]
    for t in range(T-1):
        s = np.random.choice(np.arange(n), p=P[:, s])
        states.append(s)

    i = 0
    while i < T:
        states[i] = np.nan
        i += 3

    model = FeatureDependentMarkovChain(n, n_iter=1)
    model.fit(states, np.zeros((T, 1)), verbose=False)
    Phat1 = np.array([softmax(model.models[i][1]) for i in range(n)]).T
    model = FeatureDependentMarkovChain(n, n_iter=20)
    model.fit(states, np.zeros((T, 1)), verbose=True)
    Phat2 = np.array([softmax(model.models[i][1]) for i in range(n)]).T
    print(Phat1)
    print(Phat2)
    print(P)

    print(np.abs(Phat1 - P).sum())
    print(np.abs(Phat2 - P).sum())