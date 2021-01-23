from model import FeatureDependentMarkovChain
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

if __name__ == "__main__":
    x = []
    for _ in range(100):
        T = 200
        n = 3
        P = np.random.rand(n, n) + .1
        P /= P.sum(axis=0)

        s = 0
        states = [s]
        for t in range(T-1):
            s = np.random.choice(np.arange(n), p=P[:, s])
            states.append(s)

        i = 5
        while i < T-3:
            states[i] = np.nan
            i += 3

        s = 0
        states_test = [s]
        for t in range(T-1):
            s = np.random.choice(np.arange(n), p=P[:, s])
            states_test.append(s)

        model1 = FeatureDependentMarkovChain(n, n_iter=1)
        model1.fit(states, np.zeros((T, 1)), [T], verbose=False)
        Phat1 = model1.predict(np.zeros((1, 1)))[0]
        model2 = FeatureDependentMarkovChain(n, n_iter=20)
        model2.fit(states, np.zeros((T, 1)), [T], verbose=False)
        Phat2 = model2.predict(np.zeros((1, 1)))[0]
        x.append(model2.score(states_test, np.zeros((T, 1)), [T]) - model1.score(states_test, np.zeros((T, 1)), [T]))

        print(Phat1)
        print(Phat2)
        print(P)
    print(np.min(x), np.max(x), np.mean(x), np.median(x))
    plt.hist(x, bins=50)
    plt.show()
