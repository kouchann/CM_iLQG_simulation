import numpy as np
from .autodiff import grad, hessian

class AutoDiffCost:
    """
    数値差分で l_x, l_u, l_xx, l_uu, l_ux, および終端の対応を生成。
    l(x,u) と phi(x) はユーザー定義のスカラー関数。
    """
    def __init__(self, l_func, phi_func, dt, x0):
        self.l_func = l_func
        self.phi_func = phi_func
        self.dt = dt
        self.x0 = x0

    # running terms
    def _l_scalar(self, x, u):
        return float(self.l_func(x, u)) * self.dt

    def lx(self, x, u):
        return grad(lambda xx: self._l_scalar(xx, u), x)

    def lu(self, x, u):
        return grad(lambda uu: self._l_scalar(x, uu), u)

    def lxx(self, x, u):
        return hessian(lambda xx: self._l_scalar(xx, u), x)

    def luu(self, x, u):
        return hessian(lambda uu: self._l_scalar(x, uu), u)

    def lux(self, x, u):
        # mixed second derivative via block Hessian
        def both(z):
            nx = x.size
            return self._l_scalar(z[:nx], z[nx:])
        z = np.concatenate([x, u])
        H = hessian(both, z)
        nx = x.size
        return H[nx:, :nx]  # d^2 l / du dx

    # terminal terms
    def lx_T(self, x):
        return grad(self.phi_func, x)

    def lxx_T(self, x):
        return hessian(self.phi_func, x)

    def total_cost(self, X, U):
        J = 0.0
        for k in range(len(U)):
            J += self._l_scalar(X[k], U[k])
        J += float(self.phi_func(X[-1]))
        return J
