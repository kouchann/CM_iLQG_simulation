from dataclasses import dataclass
import numpy as np

@dataclass
class ProblemParams:
    T: int                 # horizon steps
    dt: float              # step size
    nx: int                # state dim
    nu: int                # control dim
    ny: int                # obs dim

    Q: np.ndarray          # running state cost (nx,nx)
    R: np.ndarray          # running control cost (nu,nu)
    QN: np.ndarray         # terminal state cost (nx,nx)
    q: np.ndarray | None = None  # running linear term (nx,)
    r: np.ndarray | None = None  # control linear term (nu,)
    qN: np.ndarray | None = None # terminal linear term (nx,)

    Sigma_w: np.ndarray | float = 1e-4  # process noise covariance (nx,nx) or scalar
    Sigma_v: np.ndarray | float = 1e-3  # obs noise covariance (ny,ny) or scalar

    umin: np.ndarray | None = None
    umax: np.ndarray | None = None

    reg_min: float = 1e-6   # Hessian regularization lower bound
    reg_max: float = 1e+6   # Hessian regularization upper bound
    reg_init: float = 1e-3  # initial reg for backward pass
    alpha_list: tuple = (1.0, 0.5, 0.25, 0.1, 0.05)

    def expand_cov(self, M, dim):
        if np.isscalar(M): 
            return float(M) * np.eye(dim)
        return M
