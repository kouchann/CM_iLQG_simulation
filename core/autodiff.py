import numpy as np

def _central_diff_1d(func, x, i, h):
    """central difference for partial derivative wrt x[i]."""
    xp = x.copy(); xm = x.copy()
    xp[i] += h; xm[i] -= h
    return (func(xp) - func(xm)) / (2.0*h)

def jacobian(f, x, h=1e-6):
    """
    J_{ij} = d f_i / d x_j  (central difference)
    f: R^n -> R^m, x: (n,)
    """
    x = np.asarray(x, dtype=float)
    fx = np.asarray(f(x))
    m, n = fx.size, x.size
    J = np.zeros((m, n))
    for j in range(n):
        J[:, j] = _central_diff_1d(f, x, j, h).reshape(-1)
    return J

def jacobian_xy(f, x, y, which="x", h=1e-6):
    """
    Jx = df/dx or Ju = df/du for f(x,y)
    - which in {"x","y"}
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    def g_x(x_): return np.asarray(f(x_, y))
    def g_y(y_): return np.asarray(f(x, y_))
    if which == "x":
        return jacobian(g_x, x, h=h)
    else:
        return jacobian(g_y, y, h=h)

def hessian(scalar_f, x, h=1e-4):
    """
    H_{ij} = d^2 f / (dx_i dx_j)   (central difference)
    scalar_f: R^n -> R
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n))
    # diagonal and off-diagonal
    for i in range(n):
        ei = np.zeros(n); ei[i] = h
        f_pp = scalar_f(x + ei)
        f_mm = scalar_f(x - ei)
        f_00 = scalar_f(x)
        H[i, i] = (f_pp - 2*f_00 + f_mm) / (h**2)
        for j in range(i+1, n):
            ej = np.zeros(n); ej[j] = h
            f_pp = scalar_f(x + ei + ej)
            f_pm = scalar_f(x + ei - ej)
            f_mp = scalar_f(x - ei + ej)
            f_mm = scalar_f(x - ei - ej)
            val = (f_pp - f_pm - f_mp + f_mm) / (4*h*h)
            H[i, j] = val; H[j, i] = val
    return H

def grad(scalar_f, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    n = x.size
    g = np.zeros(n)
    for i in range(n):
        ei = np.zeros(n); ei[i] = h
        g[i] = (scalar_f(x + ei) - scalar_f(x - ei)) / (2*h)
    return g
