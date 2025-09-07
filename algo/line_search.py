import numpy as np

def backtracking(cost_fn, X, U, d_u_ff, d_u_fb, Xh, alphas, umin=None, umax=None):
    """
    Try step sizes alpha; return first that improves cost.
    U_new[k] = U[k] + alpha * (d_u_ff[k] + d_u_fb[k] @ (Xh[k] - X[k]))
    """
    J0 = cost_fn._cost_of(X, U)
    for a in alphas:
        U_try = []
        for k in range(len(U)):
            du = d_u_ff[k] * a + d_u_fb[k] @ (Xh[k] - X[k])
            u_new = U[k] + du
            if umin is not None: u_new = np.maximum(u_new, umin)
            if umax is not None: u_new = np.minimum(u_new, umax)
            U_try.append(u_new)
        U_try = np.array(U_try)
        J_try, X_try, _ = cost_fn.rollout_for_cost(U_try, return_traj=True)
        if J_try < J0:
            return U_try, X_try, J_try, a
    return None, None, None, None
