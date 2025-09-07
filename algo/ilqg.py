import numpy as np

class ILQG:
    """
    Basic iLQG solver:
    - forward rollout with estimator to get (X_true, X_hat, Y)
    - linearize/quadratize along nominal
    - backward pass: compute feedforward 'l' and feedback 'L' (regularized)
    - line search and iterate
    """
    def __init__(self, params, dyn, obs, estimator, cost, line_search, rng=None):
        self.p = params
        self.dyn = dyn
        self.obs = obs
        self.est = estimator
        self.cost = cost
        self.line_search = line_search
        self.rng = np.random.default_rng() if rng is None else rng

    # ---------- helpers for quadratization ----------
    def _finite_grad(self, f, x, eps=1e-6):
        g = np.zeros_like(x)
        fx = f(x)
        for i in range(x.size):
            xp = x.copy(); xp[i] += eps
            g[i] = (f(xp) - fx) / eps
        return g

    def _finite_hess(self, f, x, eps=1e-4):
        n = x.size
        H = np.zeros((n,n))
        fx = f(x)
        for i in range(n):
            ei = np.zeros(n); ei[i] = eps
            for j in range(i, n):
                ej = np.zeros(n); ej[j] = eps
                fij = f(x + ei + ej)
                fi  = f(x + ei)
                fj  = f(x + ej)
                H[i,j] = (fij - fi - fj + fx) / (eps**2)
                H[j,i] = H[i,j]
        return H

    # ---------- rollout with partial observation ----------
    def rollout(self, x0, U, return_all=False):
        # generate one noisy rollout and observations from true system
        X = [x0]
        Y = []
        for k in range(len(U)):
            x_next = self.dyn.step(X[-1], U[k], rng=self.rng)
            y_k = self.obs.measure(X[-1], U[k], rng=self.rng)
            X.append(x_next); Y.append(y_k)
        X = np.array(X); Y = np.array(Y)
        # run estimator on same inputs/observations (noisy)
        Xh, Kseq = self.est.forward(xhat0=X[0], P0=1e-3*np.eye(x0.size), U=U, Y=Y, 
                                    X_lin=X, U_lin=U)
        if return_all:
            return X, Xh, Y, Kseq
        return X, Xh

    # cost wrapper for line search
    def _cost_of(self, X, U):
        return self.cost.total_cost(X, U)

    def rollout_for_cost(self, U, return_traj=False):
        X, Xh, Y, _ = self.rollout(self.cost.x0, U, return_all=True)
        J = self._cost_of(X, U)
        return (J, X, Xh) if return_traj else J

    # expose to line_search
    def __getattr__(self, name):
        if name == "rollout":
            # already defined; line_search expects cost_fn.rollout()
            return type("RolloutAdapter",(object,),{"__call__":self.rollout_for_cost})()
        return super().__getattribute__(name)

    # ---------- backward pass ----------
    def backward_pass(self, Xh, U, reg):
        """
        Compute feedforward (kappa) and feedback (K) controls.
        Regularize Q_uu by adding reg * I to keep PD (cf. p.8).
        """
        T, nx, nu = len(U), self.p.nx, self.p.nu
        V_x = self.cost.lx_T(Xh[-1])     # d terminal V / dx
        V_xx = self.cost.lxx_T(Xh[-1])   # d2 terminal V / dxdx

        k_seq = []
        K_seq = []

        for k in reversed(range(T)):
            x = Xh[k]
            u = U[k]

            A, B = self.dyn.linearize(x, u)
            # running cost quadratization
            lx = self.cost.lx(x, u)
            lu = self.cost.lu(x, u)
            lxx = self.cost.lxx(x, u)
            luu = self.cost.luu(x, u)
            lux = self.cost.lux(x, u)

            # Q terms
            Q_x = lx + A.T @ V_x
            Q_u = lu + B.T @ V_x
            Q_xx = lxx + A.T @ V_xx @ A
            Q_ux = lux + B.T @ V_xx @ A
            Q_uu = luu + B.T @ V_xx @ B

            # regularize to ensure PD
            Q_uu_reg = Q_uu + reg * np.eye(nu)

            try:
                inv_Q_uu = np.linalg.inv(Q_uu_reg)
            except np.linalg.LinAlgError:
                return None, None, None  # signal failure

            kappa = -inv_Q_uu @ Q_u                # feedforward
            K = -inv_Q_uu @ Q_ux                   # feedback

            # value function update
            V_x = Q_x + K.T @ Q_uu @ kappa + K.T @ Q_u + Q_ux.T @ kappa
            V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
            # symmetrize
            V_xx = 0.5 * (V_xx + V_xx.T)

            k_seq.insert(0, kappa)
            K_seq.insert(0, K)

        return np.array(k_seq), np.array(K_seq), V_xx

    # ---------- main solve ----------
    def solve(self, x0, U_init, max_iter=50, tol=1e-4, verbose=True):
        U = U_init.copy()
        reg = self.p.reg_init

        J, X, Xh = self.rollout_for_cost(U, return_traj=True)

        for it in range(max_iter):
            # backward
            out = self.backward_pass(Xh, U, reg)
            if out[0] is None:
                reg = min(self.p.reg_max, reg * 10.0)
                if verbose: print(f"[it {it}] backward failed; increase reg -> {reg}")
                continue
            k_ff, K_fb, _ = out

            # line search
            from .line_search import backtracking
            U_new, X_new, J_new, alpha = backtracking(self, X, U, k_ff, K_fb, Xh,
                                                      self.p.alpha_list, self.p.umin, self.p.umax)
            if U_new is None:
                reg = min(self.p.reg_max, reg * 10.0)
                if verbose: print(f"[it {it}] no step improved cost; reg -> {reg}")
                continue

            # accept
            dJ = J - J_new
            if verbose:
                print(f"[it {it}] J: {J_new:.6e} (ΔJ={dJ:.3e}, α={alpha}, reg={reg:.1e})")
            U, X, J = U_new, X_new, J_new

            # decrease reg if things go well
            reg = max(self.p.reg_min, reg / 5.0)

            if abs(dJ) < tol:
                if verbose: print(f"Converged at iter {it}.")
                break

        # final re-run with estimator to get Xh and gains (and optionally freeze them)
        X, Xh, Y, Kseq = self.rollout(self.cost.x0, U, return_all=True)
        return {"U": U, "X": X, "Xh": Xh, "J": J, "K": Kseq}
