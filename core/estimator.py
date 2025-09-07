import numpy as np

class Estimator:
    """
    Time-varying EKF (can be 'frozen' to fixed gains after one rollout).
    Matches the non-adaptive linear filter spirit in ILQG (p.4 式(14), p.9 式(49)-(50)).
    """
    def __init__(self, dyn, obs, Sigma_w, Sigma_v, fixed_gain=False):
        self.dyn = dyn
        self.obs = obs
        self.Sigma_w = Sigma_w
        self.Sigma_v = Sigma_v
        self.fixed_gain = fixed_gain
        self.K_seq = None  # if fixed_gain=True, reuse

    def forward(self, xhat0, P0, U, Y, X_lin=None, U_lin=None):
        nx = xhat0.size
        T = len(U)
        XH = [xhat0]
        P = P0.copy()
        Ks = []

        for k in range(T):
            xlin = X_lin[k] if X_lin is not None else XH[-1]
            ulin = U_lin[k] if U_lin is not None else U[k]

            # linearize
            A, B = self.dyn.linearize(xlin, ulin)
            C, D = self.obs.linearize(xlin, ulin)
            Q = self.Sigma_w if not callable(self.Sigma_w) else self.Sigma_w(xlin, ulin)
            R = self.Sigma_v if not callable(self.Sigma_v) else self.Sigma_v(xlin, ulin)

            # predict
            x_pred = self.dyn.f(XH[-1], U[k], self.dyn.dt)
            P_pred = A @ P @ A.T + Q

            # gain
            if self.fixed_gain and self.K_seq is not None:
                K = self.K_seq[k]
            else:
                S = C @ P_pred @ C.T + R
                K = P_pred @ C.T @ np.linalg.inv(S)

            # update
            y_pred = self.obs.h(x_pred, U[k])
            x_upd = x_pred + K @ (Y[k] - y_pred)
            P_upd = (np.eye(nx) - K @ C) @ P_pred

            XH.append(x_upd)
            P = P_upd
            Ks.append(K)

        if self.fixed_gain and self.K_seq is None:
            self.K_seq = Ks

        return np.array(XH), Ks
