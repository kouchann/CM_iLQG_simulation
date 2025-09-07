import numpy as np
from typing import Callable
from .autodiff import jacobian_xy

class Dynamics:
    """
    x_{k+1} = f(x_k, u_k, dt) + w_k
    ユーザーは f だけ実装すればよく、A=df/dx, B=df/du は数値差分で自動生成。
    """
    def __init__(self, f: Callable, dt: float, Sigma_w):
        self.f = f
        self.dt = dt
        self.Sigma_w = Sigma_w

    def step(self, x, u, rng=None):
        w = np.zeros_like(x)
        if self.Sigma_w is not None:
            cov = self.Sigma_w(x, u) if callable(self.Sigma_w) else self.Sigma_w
            L = np.linalg.cholesky(cov)
            rnd = np.random.randn(x.size) if rng is None else rng.standard_normal(x.size)
            w = L @ rnd
        return self.f(x, u, self.dt) + w

    def linearize(self, x, u):
        A = jacobian_xy(lambda xx, uu: self.f(xx, uu, self.dt), x, u, which="x")
        B = jacobian_xy(lambda xx, uu: self.f(xx, uu, self.dt), x, u, which="y")
        return A, B
