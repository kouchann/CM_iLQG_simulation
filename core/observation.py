import numpy as np
from typing import Callable
from .autodiff import jacobian_xy

class Observation:
    """
    y_k = h(x_k, u_k) + v_k
    ユーザーは h だけ実装、C=dh/dx, D=dh/du は数値差分で自動生成。
    """
    def __init__(self, h: Callable, Sigma_v):
        self.h = h
        self.Sigma_v = Sigma_v

    def measure(self, x, u, rng=None):
        y = self.h(x, u)
        v = np.zeros_like(y)
        if self.Sigma_v is not None:
            cov = self.Sigma_v(x, u) if callable(self.Sigma_v) else self.Sigma_v
            L = np.linalg.cholesky(cov)
            rnd = np.random.randn(y.size) if rng is None else rng.standard_normal(y.size)
            v = L @ rnd
        return y + v

    def linearize(self, x, u):
        C = jacobian_xy(lambda xx, uu: self.h(xx, uu), x, u, which="x")
        D = jacobian_xy(lambda xx, uu: self.h(xx, uu), x, u, which="y")
        return C, D
