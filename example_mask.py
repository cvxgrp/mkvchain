from model import FeatureDependentMarkovChain
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import networkx as nx

np.set_printoptions(precision=3, suppress=True)

if __name__ == "__main__":
    T = 200
    n = 3
    P = np.random.rand(n, n) + .1
    P[0,1] = 0.
    P[2,2] = 0.
    P /= P.sum(axis=1)[:,None]


    mask = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])

    s = 0
    states = [s]
    for t in range(T-1):
        s = np.random.choice(np.arange(n), p=P[s, :])
        states.append(s)

    i = 5
    while i < T-3:
        states[i] = np.nan
        i += 3

    s = 0
    states_test = [s]
    for t in range(T-1):
        s = np.random.choice(np.arange(n), p=P[s, :])
        states_test.append(s)

    # model1 = FeatureDependentMarkovChain(n, n_iter=1, mask=mask)
    # model1.fit(states, np.zeros((T, 1)), [T], verbose=True)
    # Phat1 = model1.predict(np.zeros((1, 1)))[0]

    # print(Phat1)

    # model2 = FeatureDependentMarkovChain(n, n_iter=20, mask=mask)
    # model2.fit(states, np.zeros((T, 1)), [T], verbose=True)
    # Phat2 = model2.predict(np.zeros((1, 1)))[0]

    # print(Phat2)

    G = nx.cycle_graph(3)
    W = nx.adjacency_matrix(G) * .1

    G = nx.cycle_graph(2)
    W2 = nx.adjacency_matrix(G) * 1e4

    model2 = FeatureDependentMarkovChain(n, n_iter=20, mask=mask, W_lap_states=W, W_lap_features=W2)
    model2.fit(states, np.random.randn(T, 2), [T], verbose=True)
    Phat2 = model2.predict(np.zeros((1, 2)))[0]

    print(model2.As)