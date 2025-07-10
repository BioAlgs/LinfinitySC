from cvxopt import matrix, solvers
## K oriented
def solve_w(Y1, Y0, alpha, K, method):
    """
    Solves for weights using various optimization methods based on L1, L2, and L-Infinity norms.
    
    Args:
        Y1 (np.ndarray): The target variable array, should be a 1D or 2D array.
        Y0 (np.ndarray): The predictor variables array, should be a 2D array.
        alpha (float): The regularization parameter, relevant for methods involving regularization.
        K (float): The constraint parameter that varies based on the method.
        method (str): The optimization method to use. Options include 'sum-l2', 'sum-inf', 
                      'l1', 'l2', 'inf', 'l1-inf', 'l1-l2'.
    
    Returns:
        solution (dict): A dictionary containing the solution of the optimization problem. 
                         The keys include 'status' to indicate if the solution was found to be optimal, 
                         and other keys relevant to the optimization output.
    
    Raises:
        ValueError: If an unrecognized method is specified.
    
    Notes:
        This function configures and solves a quadratic programming problem tailored to the method specified.
        The `solvers` object referenced should be from the `cvxopt` library.
    """
    solvers.options['show_progress'] = False
    solvers.options['feastol'] = 1e-8
    if Y1.ndim == 2:
        Y1 = Y1.flatten()
    J = Y0.shape[1]
    Y0_plus = np.hstack([np.ones((Y1.shape[0], 1)), Y0]) 
    
    if method == 'sum-l2':
        
        P = Y0_plus.T @ Y0_plus
        q = -Y1.T @ Y0_plus
        G1 = np.hstack([np.zeros((J, 1)), -np.eye(J)]) # positive constraint
        G2 = np.zeros(J+1) # placeholder for l2 constraint
        G3 = np.diag([0.0] + [1.0]*J)
        G = np.vstack([G1, G2, G3])
        h = np.array([0] * J + [K] + [0] * (J+1))
        dims = {'l': J, 'q': [J+2], 's': []}
        A = np.hstack([np.zeros((1,1)), np.ones((1,J))])
        b = 1.0
        
    elif method == 'sum-inf':

        P = np.zeros((J+2, J+2))
        P[:(J+1),:(J+1)] = Y0_plus.T @ Y0_plus
        q = np.zeros(J+2)
        q[:(J+1)] = -Y1.T @ Y0_plus
        G1 = np.hstack([np.zeros((J,1)), np.eye(J), -np.ones((J,1))]) # wi - t <= 0
        G2 = np.hstack([np.zeros((J,1)), -np.eye(J), -np.ones((J,1))]) # - wi - t <= 0
        G3 = np.hstack([np.zeros((1,J+1)), -np.ones((1,1))]) # t >= 0
        G4 = np.hstack([np.zeros((J,1)), -np.eye(J), np.zeros((J,1))]) # wi >= 0
        G5 = np.hstack([np.zeros((1,J+1)), np.ones((1,1))]) # t <= K
        G = np.vstack([G1, G2, G3, G4, G5])
        h = np.append(np.zeros(G.shape[0] - 1), K)
        A = np.hstack([np.zeros((1,1)), np.ones((1,J)), np.zeros((1,1))])
        b = 1.0

    elif method == 'l1':

        P = np.zeros((2*J+1, 2*J+1))
        P[:(J+1),:(J+1)] = Y0_plus.T @ Y0_plus
        q = np.zeros(2*J+1)
        q[:(J+1)] = -Y1.T @ Y0_plus
        G1 = np.hstack([np.zeros((J,1)), np.eye(J), -np.eye(J)]) # wi - ui <= 0
        G2 = np.hstack([np.zeros((J,1)), -np.eye(J), -np.eye(J)]) # - wi - ui <= 0
        G3 = np.hstack([np.zeros((J,J+1)), -np.eye(J)]) # ui >= 0
        G4 = np.hstack([np.zeros((1,J+1)), np.ones((1, J))]) # sum ui <= K
        G = np.vstack([G1, G2, G3, G4])
        h = np.append(np.zeros(G.shape[0] - 1), K)

    elif method == 'l2':
        
        P = Y0_plus.T @ Y0_plus
        q = -Y1.T @ Y0_plus
        G1 = np.zeros(J+1) # placeholder for l2 constraint
        G2 = np.diag([0.0] + [1.0]*J) # SOC constraint
        G = np.vstack([G1, G2])
        h = np.array([K] + [0] * (J+1))
        dims = {'l': 0, 'q': [J+2], 's': []}
        
    elif method == 'inf':
        
        P = np.zeros((J+2, J+2))
        P[:(J+1),:(J+1)] = Y0_plus.T @ Y0_plus
        q = np.zeros(J+2)
        q[:(J+1)] = -Y1.T @ Y0_plus
        G1 = np.hstack([np.zeros((J,1)), np.eye(J), -np.ones((J,1))]) # wi - t <= 0
        G2 = np.hstack([np.zeros((J,1)), -np.eye(J), -np.ones((J,1))]) # - wi - t <= 0
        G3 = np.hstack([np.zeros((1,J+1)), -np.ones((1,1))]) # t >= 0
        G4 = np.hstack([np.zeros((1,J+1)), np.ones((1,1))]) # t <= K
        G = np.vstack([G1, G2, G3, G4])
        h = np.append(np.zeros(G.shape[0] - 1), K)
        
    elif method == 'l1-inf':
    
        P = np.zeros((2*J+2, 2*J+2))
        P[:(J+1),:(J+1)] = Y0_plus.T @ Y0_plus
        q = np.zeros(2*J+2)
        q[:(J+1)] = -Y1.T @ Y0_plus
        G1 = np.hstack([np.zeros((J,1)), np.eye(J), -np.eye(J), np.zeros((J,1))]) # wi - ui <= 0
        G2 = np.hstack([np.zeros((J,1)), -np.eye(J), -np.eye(J), np.zeros((J,1))]) # - wi - ui <= 0
        G3 = np.hstack([np.zeros((J,1)), np.eye(J), np.zeros((J,J)), -np.ones((J,1))]) # wi - t <= 0
        G4 = np.hstack([np.zeros((J,1)), -np.eye(J), np.zeros((J,J)), -np.ones((J,1))]) # - wi - t <= 0
        G5 = np.hstack([np.zeros((J,J+1)), -np.eye(J), np.zeros((J,1))]) # ui >= 0
        G6 = np.hstack([np.zeros((1,2*J+1)), -np.ones((1,1))]) # t >= 0
        # G7 = np.hstack([np.zeros((1,J+1)), np.ones((1, J)), alpha*np.ones((1, 1))]) # sum ui + alpha * t <= K
        G7 = np.hstack([np.zeros((1,J+1)), alpha*np.ones((1, J)), (1-alpha)*np.ones((1, 1))]) # alpha sum ui + (1-alpha) * t <= K
        G = np.vstack([G1, G2, G3, G4, G5, G6, G7])
        h = np.append(np.zeros(G.shape[0] - 1), K)

    elif method == 'l1-l2':

        P = np.zeros((2*J+1, 2*J+1))
        P[:(J+1),:(J+1)] = Y0_plus.T @ Y0_plus
        q = np.zeros(2*J+1)
        q[:(J+1)] = -Y1.T @ Y0_plus
        G1 = np.hstack([np.zeros((J,1)), np.eye(J), -np.eye(J)]) # wi - ui <= 0
        G2 = np.hstack([np.zeros((J,1)), -np.eye(J), -np.eye(J)]) # - wi - ui <= 0
        G3 = np.hstack([np.zeros((J,J+1)), -np.eye(J)]) # ui >= 0
        # G4 = np.hstack([np.zeros((1,J+1)), np.ones((1,J))]) # l1 + l2 constraint
        # G5 = np.diag([0.0] + [alpha] * J + [0.0] * J) # SOC constraint
        G4 = np.hstack([np.zeros((1,J+1)), alpha*np.ones((1,J))]) # l1 + l2 constraint
        G5 = np.diag([0.0] + [1-alpha] * J + [0.0] * J) # SOC constraint
        G = np.vstack([G1, G2, G3, G4, G5])
        h = np.array([0] * 3*J + [K] + [0] * (2*J+1))
        dims = {'l': 3*J, 'q': [2*J+2], 's': []}
        
    else:
        raise ValueError(f'No method called {method}!')
    # Convert all parameters to cvxopt matrices
    P, q, G, h = map(matrix, [P, q, G, h])
    if method in ['sum-l2', 'sum-inf']:
        A = matrix(A)
        b = matrix(b)
    else:
        A = None
        b = None

    if method in ['l2', 'sum-l2', 'l1-l2']:
        solution = solvers.coneqp(P, q, G, h, dims, A, b)
    else:    
        # Solve the quadratic programming problem
        solution = solvers.qp(P, q, G, h, A, b)
    if solution['status'] != 'optimal':
        warnings.warn(f'Optimal solution not found for {method}!')
    return solution


## lambda oriented
def solve_w(Y1, Y0, alpha, lam, method):
    solvers.options['show_progress'] = False
    if Y1.ndim == 2:
        Y1 = Y1.flatten()
    J = Y0.shape[1]
    Y0_plus = np.hstack([np.ones((Y1.shape[0], 1)), Y0]) 
    
    if method == 'sum-l2':
        
        P = Y0_plus.T @ Y0_plus + lam * np.diag([0] + [1]*(J))
        q = -Y1.T @ Y0_plus
        G = np.hstack([np.zeros((J,1)), -np.eye(J)]) # wi >= 0
        h = np.zeros(G.shape[0])
        A = np.hstack([np.zeros((1,1)), np.ones((1,J))])
        b = 1.0

    elif method == 'sum-inf':

        P = np.zeros((J+2, J+2))
        P[:(J+1),:(J+1)] = Y0_plus.T @ Y0_plus
        q = np.zeros(J+2)
        q[:(J+1)] = -Y1.T @ Y0_plus
        q[J+1] = lam
        G1 = np.hstack([np.zeros((J,1)), np.eye(J), -np.ones((J,1))]) # wi - t <= 0
        G2 = np.hstack([np.zeros((J,1)), -np.eye(J), -np.ones((J,1))]) # - wi - t <= 0
        G3 = np.hstack([np.zeros((1,J+1)), -np.ones((1,1))]) # t >= 0
        G4 = np.hstack([np.zeros((J,1)), -np.eye(J), np.zeros((J,1))]) # wi >= 0
        G = np.vstack([G1, G2, G3, G4])
        h = np.zeros(G.shape[0])
        A = np.hstack([np.zeros((1,1)), np.ones((1,J)), np.zeros((1,1))])
        b = 1.0

    elif method == 'l1':

        P = np.zeros((2*J+1, 2*J+1))
        P[:(J+1),:(J+1)] = Y0_plus.T @ Y0_plus
        q = np.zeros(2*J+1)
        q[:(J+1)] = -Y1.T @ Y0_plus
        q[(J+1):] = lam
        G1 = np.hstack([np.zeros((J,1)), np.eye(J), -np.eye(J)]) # wi - ui <= 0
        G2 = np.hstack([np.zeros((J,1)), -np.eye(J), -np.eye(J)]) # - wi - ui <= 0
        G3 = np.hstack([np.zeros((J,J+1)), -np.eye(J)]) # ui >= 0
        G = np.vstack([G1, G2, G3])
        h = np.zeros(G.shape[0])

    elif method == 'l2':
        
        P = Y0_plus.T @ Y0_plus + lam * np.diag([0] + [1]*(J))
        q = -Y1.T @ Y0_plus

        P = matrix(P)
        q = matrix(q)
        G = None
        h = None
        A = None
        b = None
        
    elif method == 'inf':
        
        P = np.zeros((J+2, J+2))
        P[:(J+1),:(J+1)] = Y0_plus.T @ Y0_plus
        q = np.zeros(J+2)
        q[:(J+1)] = -Y1.T @ Y0_plus
        q[J+1] = lam
        G1 = np.hstack([np.zeros((J,1)), np.eye(J), -np.ones((J,1))]) # wi - t <= 0
        G2 = np.hstack([np.zeros((J,1)), -np.eye(J), -np.ones((J,1))]) # - wi - t <= 0
        G3 = np.hstack([np.zeros((1,J+1)), -np.ones((1,1))]) # t >= 0
        G = np.vstack([G1, G2, G3])
        h = np.zeros(G.shape[0])
        
    elif method == 'l1-inf':
    
        P = np.zeros((2*J+2, 2*J+2))
        P[:(J+1),:(J+1)] = Y0_plus.T @ Y0_plus
        q = np.zeros(2*J+2)
        q[:(J+1)] = -Y1.T @ Y0_plus
        q[(J+1):(2*J+1)] = lam
        q[2*J+1] = lam * alpha
        G1 = np.hstack([np.zeros((J,1)), np.eye(J), -np.eye(J), np.zeros((J,1))]) # wi - ui <= 0
        G2 = np.hstack([np.zeros((J,1)), -np.eye(J), -np.eye(J), np.zeros((J,1))]) # - wi - ui <= 0
        G3 = np.hstack([np.zeros((J,1)), np.eye(J), np.zeros((J,J)), -np.ones((J,1))]) # wi - t <= 0
        G4 = np.hstack([np.zeros((J,1)), -np.eye(J), np.zeros((J,J)), -np.ones((J,1))]) # - wi - t <= 0
        G5 = np.hstack([np.zeros((J,J+1)), -np.eye(J), np.zeros((J,1))]) # ui >= 0
        G6 = np.hstack([np.zeros((1,2*J+1)), -np.ones((1,1))]) # t >= 0
        G = np.vstack([G1, G2, G3, G4, G5, G6])
        h = np.zeros(G.shape[0])

    elif method == 'l1-l2':

        P = np.zeros((2*J+1, 2*J+1))
        P[:(J+1),:(J+1)] = Y0_plus.T @ Y0_plus + lam * alpha * np.diag([0] + [1]*(J))
        q = np.zeros(2*J+1)
        q[:(J+1)] = -Y1.T @ Y0_plus
        q[(J+1):] = lam
        G1 = np.hstack([np.zeros((J,1)), np.eye(J), -np.eye(J)]) # wi - ui <= 0
        G2 = np.hstack([np.zeros((J,1)), -np.eye(J), -np.eye(J)]) # - wi - ui <= 0
        G3 = np.hstack([np.zeros((J,J+1)), -np.eye(J)]) # ui >= 0
        G = np.vstack([G1, G2, G3])
        h = np.zeros(G.shape[0])

    # Convert all parameters to cvxopt matrices
    P, q = map(matrix, [P, q])
    if method == 'l2':
        G, h = None, None
    else:
        G, h = map(matrix, [G, h])
    if method in ['sum-l2', 'sum-inf']:
        A = matrix(A)
        b = matrix(b)
    else:
        A = None
        b = None
    # Solve the quadratic programming problem
    solution = solvers.qp(P, q, G, h, A, b)
    if solution['status'] != 'optimal':
        warnings.warn(f'Optimal solution not found for {method}!')
    return solution


def solve_column(Y1, Y0, alpha, K, method, solver):
    # Return the resulting column as a numpy array
    # Define the variable w (n_0 by 1 vector)
    if Y1.ndim == 2:
        Y1 = Y1.flatten()
    J = Y0.shape[1]
    Y0_plus = np.hstack([np.ones((Y1.shape[0], 1)), Y0])
    
    # w_plus = cp.Variable(J + 1, nonneg=True)
    # w_minus = cp.Variable(J + 1, nonneg=True)
    # w = w_plus - w_minus
    # Z = cp.Variable(J, boolean=True)
    # l1 = cp.sum(w_plus[1:] + w_minus[1:])
    
    # if method == 's-const':
    #     ## LS + sum constraint
    #     constraints = [ w[1:] >= 0,
    #                     cp.sum(w[1:]) == 1 ]
    #     problem = cp.Problem(objective, constraints)
    if method == 'sum-inf':
        ## LS + sum constraint + infinity norm
        w = cp.Variable(J + 1)
        l_inf = cp.norm(w[1:], 'inf')
        constraints = [ w[1:] >= 0,
                        cp.sum(w[1:]) == 1,
                        l_inf <= K ]
    elif method == 'sum-l2':
        ## LS + sum constraint + L2
        w = cp.Variable(J + 1)
        l2sq = cp.sum_squares(w[1:])
        constraints = [ w[1:] >= 0,
                        cp.sum(w[1:]) == 1,
                        l2sq <= K ]
    elif method == 'inf':
        ## LS + infinity norm
        w = cp.Variable(J + 1)
        l_inf = cp.norm(w[1:], 'inf')
        constraints = [ l_inf <= K ]
    elif method == 'l1':
        ## LS + constrained Lasso
        w = cp.Variable(J + 1)
        l1 = cp.norm(w[1:], 1)
        constraints = [ l1 <= K ]
    elif method == 'l2':
        ## LS + L2
        w = cp.Variable(J + 1)
        l2sq = cp.sum_squares(w[1:])
        constraints = [ l2sq <= K ]
    elif method == 'l1-inf':
        ## LS + L1 + infinity norm
        w = cp.Variable(J + 1)
        l1 = cp.norm(w[1:], 1)
        l_inf = cp.norm(w[1:], 'inf')
        constraints = [ l1 + alpha * l_inf <= K ]
    elif method == 'l1-l2':
        ## LS + L1 + L2
        w = cp.Variable(J + 1)
        l1 = cp.norm(w[1:], 1)
        l2sq = cp.sum_squares(w[1:])
        constraints = [ l1 + alpha * l2sq <= K ]

    
    # P = Y0_plus.T @ Y0_plus  # This forms the quadratic part
    # Pp = cp.Parameter(shape=P.shape, value=P, PSD=True)
    # q = -2 * Y1.T @ Y0_plus  # This forms the linear part
    # objective = cp.Minimize(cp.quad_form(w, Pp) + q @ w)
    loss = cp.norm(Y1 - cp.matmul(Y0_plus, w), 2)**2
    objective = cp.Minimize(loss)
    problem = cp.Problem(objective, constraints)
    # Solve the problem
    if solver == 'CLARABEL':
        problem.solve(solver=cp.CLARABEL)
    elif solver == 'CPLEX':
        # problem.solve(solver=cp.CPLEX,
        #               cplex_params={
        #                   "timelimit": 60,  # 5 seconds
        #                   "mip.limits.nodes": 1000, 
        #                   "simplex.limits.iterations": 5000
        #               })
        problem.solve(solver=cp.CPLEX)
    elif solver == 'CVXOPT':
        problem.solve(solver=cp.CVXOPT)
    elif solver == 'ECOS':
        problem.solve(solver=cp.ECOS)
    elif solver == 'GUROBI':
        problem.solve(solver=cp.GUROBI)
    elif solver == 'SCIP':
        problem.solve(solver=cp.SCIP)
    elif solver == 'SCS':
        problem.solve(solver=cp.SCS)
    # problem.solve()
    if w.value is None:
        w = np.insert(np.ones(J) / J, 0, 0)
    else:
        w = w.value
    return w

# parameter selector
# leave-one-out cross validation to find the optimal alpha value
# def loo_selector(Y0_pre, Y0_post, method, start=-5, end=1, n=15, verbose=False):
#     # if Y0_post.ndim == 1:
#     #     Y0t2 = Y0t2.reshape(-1, 1)
#     J = Y0_pre.shape[1]
#     T0 = Y0_pre.shape[0]
#     T1 = Y0_post.shape[0]
#     alpha_list = K_list = np.logspace(start, end, n) # Alpha value
#     sspe_list = np.zeros((len(alpha_list), len(K_list)))
#     for i, alpha in enumerate(alpha_list):
#         for j, K in enumerate(K_list):
#             tau = np.zeros((J, T1))
#             for idx in range(J):
#                 Ynm1_pre = np.delete(Y0_pre, idx, axis=1)
#                 w0 = solve_column(Y0_pre[:, idx], Ynm1_pre, alpha, method, verbose)
#                 Ynm1_post = np.delete(Y0_post, idx, axis=1)
#                 # print(Ynm1_post.shape)
#                 # print(w0.shape)
#                 tau[idx, :] = np.mean(Y0_post[:, idx] - Ynm1_post @ w0)
#             sspe_list[i, j] = np.sum(np.square(tau))
#     min_idx = np.unravel_index(np.argmin(sspe_list, axis=None), sspe_list.shape)
#     return alpha_list[min_idx[0]], K_list[min_idx[1]]

# latex_table = df_averages.to_latex(index=True)
# Print or write the LaTeX table to a file
# print(latex_table)