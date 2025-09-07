import numpy as np
import pickle
from core.params import ProblemParams
from core.dynamics import Dynamics
from core.observation import Observation
from core.estimator import Estimator
from core.cost import AutoDiffCost
from algo.ilqg import ILQG
from models import pendulum as pend
from plot.plot_results import plot_results


def main():
    dt = 0.01
    T = 400            # 4 seconds
    nx, nu, ny = 2, 1, 1
    x0 = np.array([np.pi, 0.0])     # 逆さ（下）から立ち上げ
    x_goal = np.array([0.0, 0.0])

    Q  = np.diag([5.0, 0.1])
    R  = np.diag([0.05])
    QN = np.diag([30.0, 1.0])

    Sigma_w = 1e-4 * np.eye(nx)
    Sigma_v = 1e-3 * np.eye(ny)

    params = ProblemParams(T=T, dt=dt, nx=nx, nu=nu, ny=ny,
                           Q=Q, R=R, QN=QN,
                           Sigma_w=Sigma_w, Sigma_v=Sigma_v,
                           umin=np.array([-5.0]), umax=np.array([5.0]))

    dyn = Dynamics(f=pend.f, dt=dt, Sigma_w=Sigma_w)
    obs = Observation(h=pend.h, Sigma_v=Sigma_v)

    est = Estimator(dyn, obs, Sigma_w, Sigma_v, fixed_gain=False)

    # 自動微分版コスト（数値差分）
    l, phi = pend.make_quadratic_cost(x_goal, Q, R, QN)
    cost = AutoDiffCost(l_func=l, phi_func=phi, dt=dt, x0=x0)

    U_init = np.zeros((T, nu))
    solver = ILQG(params, dyn, obs, est, cost, line_search=None)

    result = solver.solve(x0, U_init, max_iter=60, tol=1e-5, verbose=True)

    print("\nFinal cost:", result["J"])
    print("First 5 controls:", result["U"][:5].ravel())
    print("Terminal state (true):", result["X"][-1])
    print("Terminal state (est):", result["Xh"][-1])

    with open("ilqg_result.pkl", "wb") as f:
        pickle.dump(result, f)
    plot_results(result)


if __name__ == "__main__":
    main()
