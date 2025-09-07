import numpy as np

# --- 非線形力学 f(x,u,dt) ---
def f(x, u, dt):
    # x = [theta, omega]
    g, L, b, m = 9.81, 1.0, 0.05, 1.0
    th, w = x
    tau = u[0]
    th_dot = w
    w_dot = -(g/L)*np.sin(th) - b*w + tau/(m*L*L)
    xdot = np.array([th_dot, w_dot])
    return x + dt * xdot

# --- 観測 h(x,u)（角度のみ観測） ---
def h(x, u):
    return np.array([x[0]])

# --- ステージコスト l(x,u), 終端コスト phi(x)（二次型だが数式は不要） ---
def make_quadratic_cost(x_goal, Q, R, QN):
    def l(x, u):
        dx = x - x_goal
        return 0.5 * (dx @ Q @ dx + u @ R @ u)
    def phi(x):
        dx = x - x_goal
        return 0.5 * (dx @ QN @ dx)
    return l, phi
