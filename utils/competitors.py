import numpy as np
import cvxpy as cp
from cvxopt import matrix, solvers

# Difference-in-Differences
def did(Y1, Y0):
    theta_hat = Y1 - np.mean(Y1 - np.mean(Y0, axis=1, keepdims=True)) - np.mean(Y0, axis=1)
    return theta_hat

# Synthetic Control
def sc(Y1, Y0, scale=False):
    solvers.options['show_progress'] = False
    if scale:
        Y0_sd = np.std(Y0, axis=0)
        Y0 = Y0 / Y0_sd
    J = Y0.shape[1]

    # Quadratic term (P) and linear term (q) for the objective function
    P = matrix(Y0.T @ Y0)
    q = matrix(-Y0.T @ Y1)
    # Equality constraint: Sum of weights equals 1
    A = matrix(1.0, (1, J))
    b = matrix(1.0)
    # Inequality constraint: All weights are non-negative
    G = matrix(-np.eye(J))
    h = matrix(np.zeros(J))

    # Solving the quadratic program
    sol = solvers.qp(P, q, G, h, A, b)
    w_hat = np.array(sol['x']).flatten()
    if scale:
        w_hat = w_hat / Y0_sd
    # u_hat = Y1 - Y0 @ w_hat
    # return {'u_hat': u_hat, 'w_hat': w_hat}
    return w_hat

# Constrained Lasso
def classo(Y1, Y0, K):
    J = Y0.shape[1]
    w = cp.Variable(J+1)
    Y1 = Y1.flatten()
    # print(Y1.shape)
    # print(Y0.shape)
    loss = cp.sum_squares(Y1 - cp.hstack([np.ones((Y1.shape[0], 1)), Y0]) @ w) / Y1.shape[0]
    constr = [cp.norm(w[1:], 1) <= K]
    prob = cp.Problem(cp.Minimize(loss), constr)
    prob.solve()
    w_hat = w.value
    # u_hat = Y1 - np.hstack([np.ones((Y1.shape[0], 1)), Y0]) @ w_hat
    # return {'u_hat': u_hat, 'w_hat': w_hat}
    return w_hat