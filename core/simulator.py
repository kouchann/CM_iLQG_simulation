import numpy as np

class Simulator:
    def __init__(self, dyn, obs):
        self.dyn = dyn
        self.obs = obs

    def propagate_true(self, x0, U, rng=None):
        X = [x0]
        for u in U:
            x_next = self.dyn.step(X[-1], u, rng=rng)
            X.append(x_next)
        return np.array(X)  # (T+1,nx)

    def observe(self, X, U, rng=None):
        Y = []
        for k in range(len(U)):
            y = self.obs.measure(X[k], U[k], rng=rng)
            Y.append(y)
        return np.array(Y)  # (T,ny)
