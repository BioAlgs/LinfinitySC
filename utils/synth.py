import numpy as np
import warnings
from cvxopt import matrix, solvers
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split, KFold 

solvers.options['show_progress'] = False
solvers.options['abstol'] = 1e-9
solvers.options['reltol'] = 1e-8
solvers.options['feastol'] = 1e-8

## lambda oriented
def solve_w(Y1, Y0, alpha, lam, method, std):

    if Y1.ndim == 2:
        Y1 = Y1.flatten()
    if std:
        Y0 = Y0 / np.std(Y0, axis=0)
    T, J = Y0.shape
    Y0_plus = np.hstack([np.ones((T, 1)), Y0]) 
    
    if method == 'sum-l2':
        
        P = 1/(T) * Y0_plus.T @ Y0_plus + lam * np.diag([0] + [1]*(J))
        q = -1/(T) * Y1.T @ Y0_plus
        G = np.hstack([np.zeros((J,1)), -np.eye(J)]) # wi >= 0
        h = np.zeros(G.shape[0])
        A = np.hstack([np.zeros((1,1)), np.ones((1,J))])
        b = 1.0

    elif method == 'sum-inf':

        P = np.zeros((J+2, J+2))
        P[:(J+1),:(J+1)] = 1/(T) * Y0_plus.T @ Y0_plus
        q = np.zeros(J+2)
        q[:(J+1)] = -1/(T) * Y1.T @ Y0_plus
        q[J+1] = lam
        G1 = np.hstack([np.zeros((J,1)), np.eye(J), -np.ones((J,1))]) # wi - t <= 0
        G2 = np.hstack([np.zeros((J,1)), -np.eye(J), -np.ones((J,1))]) # - wi - t <= 0
        # G3 = np.hstack([np.zeros((1,J+1)), -np.ones((1,1))]) # t >= 0
        G4 = np.hstack([np.zeros((J,1)), -np.eye(J), np.zeros((J,1))]) # wi >= 0
        G = np.vstack([G1, G2, G4])
        h = np.zeros(G.shape[0])
        A = np.hstack([np.zeros((1,1)), np.ones((1,J)), np.zeros((1,1))])
        b = 1.0

    elif method == 'l1':

        P = np.zeros((2*J+1, 2*J+1))
        P[:(J+1),:(J+1)] = 1/(T) * Y0_plus.T @ Y0_plus
        q = np.zeros(2*J+1)
        q[:(J+1)] = -1/(T) * Y1.T @ Y0_plus
        q[(J+1):] = lam
        G1 = np.hstack([np.zeros((J,1)), np.eye(J), -np.eye(J)]) # wi - ui <= 0
        G2 = np.hstack([np.zeros((J,1)), -np.eye(J), -np.eye(J)]) # - wi - ui <= 0
        # G3 = np.hstack([np.zeros((J,J+1)), -np.eye(J)]) # ui >= 0
        G = np.vstack([G1, G2])#, G3])
        h = np.zeros(G.shape[0])

    elif method == 'l2':

        lam = max(lam, np.std(Y1) * 1e-8)
        P = 1/(T) * Y0_plus.T @ Y0_plus + 1/(np.std(Y1)) * lam * np.diag([0] + [1]*(J))
        q = -1/(T) * Y1.T @ Y0_plus

        P = matrix(P)
        q = matrix(q)
        
    elif method == 'inf':
        
        P = np.zeros((J+2, J+2))
        P[:(J+1),:(J+1)] = 1/(T) * Y0_plus.T @ Y0_plus
        q = np.zeros(J+2)
        q[:(J+1)] = -1/(T) * Y1.T @ Y0_plus
        q[J+1] = lam
        G1 = np.hstack([np.zeros((J,1)), np.eye(J), -np.ones((J,1))]) # wi - t <= 0
        G2 = np.hstack([np.zeros((J,1)), -np.eye(J), -np.ones((J,1))]) # - wi - t <= 0
        # G3 = np.hstack([np.zeros((1,J+1)), -np.ones((1,1))]) # t >= 0
        G = np.vstack([G1, G2])#, G3])
        h = np.zeros(G.shape[0])
        
    elif method == 'l1-inf':
    
        P = np.zeros((2*J+2, 2*J+2))
        P[:(J+1),:(J+1)] = 1/(T) * Y0_plus.T @ Y0_plus
        q = np.zeros(2*J+2)
        q[:(J+1)] = -1/(T) * Y1.T @ Y0_plus
        q[(J+1):(2*J+1)] = lam * alpha
        q[2*J+1] = lam * (1 - alpha)
        G1 = np.hstack([np.zeros((J,1)), np.eye(J), -np.eye(J), np.zeros((J,1))]) # wi - ui <= 0
        G2 = np.hstack([np.zeros((J,1)), -np.eye(J), -np.eye(J), np.zeros((J,1))]) # - wi - ui <= 0
        G3 = np.hstack([np.zeros((J,1)), np.eye(J), np.zeros((J,J)), -np.ones((J,1))]) # wi - t <= 0
        G4 = np.hstack([np.zeros((J,1)), -np.eye(J), np.zeros((J,J)), -np.ones((J,1))]) # - wi - t <= 0
        # G5 = np.hstack([np.zeros((J,J+1)), -np.eye(J), np.zeros((J,1))]) # ui >= 0
        # G6 = np.hstack([np.zeros((1,2*J+1)), -np.ones((1,1))]) # t >= 0
        G = np.vstack([G1, G2, G3, G4])#, G5, G6])
        h = np.zeros(G.shape[0])

    elif method == 'l1-l2':

        P = np.zeros((2*J+1, 2*J+1))
        P[:(J+1),:(J+1)] = 1/(T) * Y0_plus.T @ Y0_plus + 1/(np.std(Y1)) * lam * (1 - alpha) * np.diag([0] + [1]*(J))
        q = np.zeros(2*J+1)
        q[:(J+1)] = -1/(T) * Y1.T @ Y0_plus
        q[(J+1):] = lam * alpha
        G1 = np.hstack([np.zeros((J,1)), np.eye(J), -np.eye(J)]) # wi - ui <= 0
        G2 = np.hstack([np.zeros((J,1)), -np.eye(J), -np.eye(J)]) # - wi - ui <= 0
        # G3 = np.hstack([np.zeros((J,J+1)), -np.eye(J)]) # ui >= 0
        G = np.vstack([G1, G2])#, G3])
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

def solve_w0(Y1, Y0, alpha, lam, method, std):
    
    if Y1.ndim == 2:
        Y1 = Y1.flatten()
    if std:
        Y0 = Y0 / np.std(Y0, axis=0)
    T, J = Y0.shape
    
    if method == 'sum-l2':
        
        P = 1/(T) * Y0.T @ Y0 + lam * np.diag([0] + [1]*(J))
        q = -1/(T) * Y1.T @ Y0
        G = np.hstack(-np.eye(J)) # wi >= 0
        h = np.zeros(G.shape[0])
        A = np.hstack(np.ones((1,J)))
        b = 1.0

    elif method == 'sum-inf':

        P = np.zeros((J+1, J+1))
        P[:J,:J] = 1/(T) * Y0.T @ Y0
        q = np.zeros(J+1)
        q[:J] = -1/(T) * Y1.T @ Y0
        q[J] = lam
        G1 = np.hstack([np.eye(J), -np.ones((J,1))]) # wi - t <= 0
        G2 = np.hstack([-np.eye(J), -np.ones((J,1))]) # - wi - t <= 0
        # G3 = np.hstack([np.zeros((1,J)), -np.ones((1,1))]) # t >= 0
        # G4 = np.hstack([-np.eye(J), np.zeros((J,1))]) # wi >= 0
        G = np.vstack([G1, G2])#, G3, G4])
        h = np.zeros(G.shape[0])
        A = np.hstack([np.ones((1,J)), np.zeros((1,1))])
        b = 1.0

    elif method == 'l1':

        P = np.zeros((2*J, 2*J))
        P[:J,:J] = 1/(T) * Y0.T @ Y0
        q = np.zeros(2*J)
        q[:J] = -1/(T) * Y1.T @ Y0
        q[J:] = lam
        G1 = np.hstack([np.eye(J), -np.eye(J)]) # wi - ui <= 0
        G2 = np.hstack([-np.eye(J), -np.eye(J)]) # - wi - ui <= 0
        # G3 = np.hstack([np.zeros((J,J)), -np.eye(J)]) # ui >= 0
        G = np.vstack([G1, G2])#, G3])
        h = np.zeros(G.shape[0])

    elif method == 'l2':

        lam = max(lam, np.std(Y1) * 1e-8)
        P = 1/(T) * Y0.T @ Y0 + 1/(np.std(Y1)) * lam * np.eye(J)
        q = -1/(T) * Y1.T @ Y0

        P = matrix(P)
        q = matrix(q)
        
    elif method == 'inf':
        
        P = np.zeros((J+1, J+1))
        P[:J,:J] = 1/(T) * Y0.T @ Y0
        q = np.zeros(J+1)
        q[:J] = -1/(T) * Y1.T @ Y0
        q[J] = lam
        G1 = np.hstack([np.eye(J), -np.ones((J,1))]) # wi - t <= 0
        G2 = np.hstack([-np.eye(J), -np.ones((J,1))]) # - wi - t <= 0
        # G3 = np.hstack([np.zeros((1,J)), -np.ones((1,1))]) # t >= 0
        G = np.vstack([G1, G2])#, G3])
        h = np.zeros(G.shape[0])
        
    elif method == 'l1-inf':
    
        P = np.zeros((2*J+1, 2*J+1))
        P[:J,:J] = 1/(T) * Y0.T @ Y0
        q = np.zeros(2*J+1)
        q[:J] = -1/(T) * Y1.T @ Y0
        q[J:(2*J)] = lam * alpha
        q[2*J] = lam * (1 - alpha)
        G1 = np.hstack([np.eye(J), -np.eye(J), np.zeros((J,1))]) # wi - ui <= 0
        G2 = np.hstack([-np.eye(J), -np.eye(J), np.zeros((J,1))]) # - wi - ui <= 0
        G3 = np.hstack([np.eye(J), np.zeros((J,J)), -np.ones((J,1))]) # wi - t <= 0
        G4 = np.hstack([-np.eye(J), np.zeros((J,J)), -np.ones((J,1))]) # - wi - t <= 0
        # G5 = np.hstack([np.zeros((J,J)), -np.eye(J), np.zeros((J,1))]) # ui >= 0
        # G6 = np.hstack([np.zeros((1,2*J)), -np.ones((1,1))]) # t >= 0
        G = np.vstack([G1, G2, G3, G4])#, G5, G6])
        h = np.zeros(G.shape[0])

    elif method == 'l1-l2':

        P = np.zeros((2*J, 2*J))
        P[:J,:J] = 1/(T) * Y0.T @ Y0 + 1/(np.std(Y1)) * lam * (1 - alpha) * np.eye(J)
        q = np.zeros(2*J)
        q[:J] = -1/(T) * Y1.T @ Y0
        q[J:] = lam * alpha
        G1 = np.hstack([np.eye(J), -np.eye(J)]) # wi - ui <= 0
        G2 = np.hstack([-np.eye(J), -np.eye(J)]) # - wi - ui <= 0
        # G3 = np.hstack([np.zeros((J,J)), -np.eye(J)]) # ui >= 0
        G = np.vstack([G1, G2])#, G3])
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

## wrapper function for our method
def our(Y1, Y0, alpha, lam, method, std=False, intercept=True):

    J = Y0.shape[1]
    if intercept:
        sol = solve_w(Y1, Y0, alpha, lam, method, std)
        vec = np.array(sol['x']).flatten()
        w_hat = vec[:(J+1)]
        if std:
            w_hat[1:] = w_hat[1:] / np.std(Y0, axis=0)
    else:
        sol = solve_w0(Y1, Y0, alpha, lam, method, std)
        vec = np.array(sol['x']).flatten()
        w_hat = vec[:J]
        if std:
            w_hat = w_hat / np.std(Y0, axis=0)

    return w_hat


# def cv_sspe(Y1_pre, Y0_pre, method, alpha, lam, std, intercept, n_folds=5):
#     n_folds = min(n_folds, len(Y0_pre))
#     kf = KFold(n_splits=n_folds, shuffle=True) 
#     Tau = np.zeros_like(Y1_pre)
#     for train_index, test_index in kf.split(Y0_pre):
#         w = our(Y1_pre[train_index], Y0_pre[train_index, :], alpha, lam, method, std, intercept)
#         if intercept:
#             Y0_pre_plus = np.hstack([np.ones((len(Y0_pre), 1)), Y0_pre]) 
#             Tau[test_index] = Y1_pre[test_index] - Y0_pre_plus[test_index] @ w
#         else:
#             Tau[test_index] = Y1_pre[test_index] - Y0_pre[test_index] @ w
#     # sspe = np.sum(np.square(np.sum(Tau, axis=0)))
#     sspe = np.sum(np.square(Tau))
#     return sspe

# def hold_out_sspe(Y1_pre, Y0_pre, method, alpha, lam, std, intercept, test_size=0.4):
#     # Split the data into training and testing sets
#     train_index, test_index = train_test_split(np.arange(len(Y0_pre)), test_size=test_size)
    
#     w = our(Y1_pre[train_index], Y0_pre[train_index, :], alpha, lam, method, std, intercept)
#     if intercept:
#         Y0_pre_plus = np.hstack([np.ones((len(Y0_pre), 1)), Y0_pre]) 
#         Tau = Y1_pre[test_index] - Y0_pre_plus[test_index] @ w
#     else:
#         Tau = Y1_pre[test_index] - Y0_pre[test_index] @ w

#     # sspe = np.sum(np.square(np.sum(Tau, axis=0)))
#     sspe = np.sum(np.square(Tau))
#     return sspe

def cv_sspe(Y1_pre, Y0_pre, method, alpha, lam, std, intercept, n_folds, n_repeats):
    
    n_folds = min(n_folds, len(Y0_pre))
    total_sspe = 0

    for r in range(n_repeats):
        # Ensure reproducibility
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=r)  # Fixed random state
        Tau = np.zeros_like(Y1_pre)
        for train_index, test_index in kf.split(Y0_pre):
            w = our(Y1_pre[train_index], Y0_pre[train_index, :], alpha, lam, method, std, intercept)
            if intercept:
                Y0_pre_plus = np.hstack([np.ones((len(Y0_pre), 1)), Y0_pre]) 
                Tau[test_index] = Y1_pre[test_index] - Y0_pre_plus[test_index] @ w
            else:
                Tau[test_index] = Y1_pre[test_index] - Y0_pre[test_index] @ w
        total_sspe += np.sum(np.square(Tau))  # Sum SSPE for this repeat

    average_sspe = total_sspe / n_repeats  # Average across repetitions
    return average_sspe

def hold_out_sspe(Y1_pre, Y0_pre, method, alpha, lam, std, intercept, test_size, n_repeats):
    
    total_sspe = 0

    for r in range(n_repeats):
        # Ensure reproducibility
        train_index, test_index = train_test_split(
            np.arange(len(Y0_pre)), test_size=test_size, random_state=r
        )
        w = our(Y1_pre[train_index], Y0_pre[train_index, :], alpha, lam, method, std, intercept)
        if intercept:
            Y0_pre_plus = np.hstack([np.ones((len(Y0_pre), 1)), Y0_pre]) 
            Tau = Y1_pre[test_index] - Y0_pre_plus[test_index] @ w
        else:
            Tau = Y1_pre[test_index] - Y0_pre[test_index] @ w
        total_sspe += np.sum(np.square(Tau))  # Sum SSPE for this repeat

    average_sspe = total_sspe / n_repeats  # Average across repetitions
    return average_sspe

def param_selector(Y1_pre, Y0_pre, method, fixed_alpha=None, fixed_lam=None, 
                          std=False, intercept=True, n_folds=None, test_size=None, 
                          n_repeats=2, max_workers=16):

    T0, J = Y0_pre.shape
    if fixed_lam is not None and fixed_alpha is not None:
        return (float(fixed_alpha), float(fixed_lam)) 
    
    # Initialize alpha_list
    if method in ['l1']:
        alpha_list = [1.0]
    elif method in ['l2']:
        alpha_list = [1.0/np.sqrt(J)] # correction factor for generate_lambda_seq
    elif method in ['inf']:
        alpha_list = [1.0/J] # correction factor for generate_lambda_seq
    elif fixed_alpha is None:
        alpha_list = np.linspace(0.0, 1.0, num=21)
        # alpha_list = np.sin(np.pi * np.linspace(0.0, 1.0, num=21) / 2) ** 2  # alpha values
    elif isinstance(fixed_alpha, (float, int)):
        alpha_list = [float(fixed_alpha)]  # Single alpha value if fixed
    else:
        raise ValueError("fixed_alpha is not correctly specified.")
    

    # Initialize an empty list to hold parameter tuples
    param_list = []
    for alpha in alpha_list:
        # Determine lam_list based on alpha
        if fixed_lam is None:
            lam_list = generate_lambda_seq(Y1_pre, Y0_pre, alpha)
        else:
            lam_list = [float(fixed_lam)]  # Single lam value if fixed
        # Append all (alpha, lam) tuples to param_list
        param_list.extend([(alpha, lam) for lam in lam_list])

    # Initialize parallel computation
    if n_folds is not None:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(cv_sspe, Y1_pre, Y0_pre, method, alpha, lam, 
                                       std, intercept, n_folds, n_repeats) for alpha, lam in param_list]
            sspe_list = [future.result() for future in futures]
    elif test_size is not None:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(hold_out_sspe, Y1_pre, Y0_pre, method, alpha, lam, 
                                       std, intercept, test_size, n_repeats) for alpha, lam in param_list]
            sspe_list = [future.result() for future in futures]
    else:
        raise ValueError("Please define at least one of n_folds or test_size.")
    
    min_index = np.argmin(sspe_list)
    best_params = param_list[min_index]
    return best_params

def generate_lambda_seq(Y1, Y0, alpha, epsilon=0.0001, num=30):
    # standardize the predictor variables
    sY0 = (Y0 - np.mean(Y0, axis=0)) / np.apply_along_axis(np.std, 0, Y0)
    alpha = max(alpha, .01)
    # Calculate lambda_max using the provided alpha
    lam_max = np.max(np.abs(sY0.T @ Y1)) / (Y0.shape[0] * alpha)
    lam_min = lam_max * epsilon

    # Make sure the range of lambda is not too off
    lam_max = min(lam_max, 20)
    lam_min = min(lam_min, 1e-4)
    lam_seq = np.exp(np.linspace(np.log(lam_max), np.log(lam_min), num))
    
    return lam_seq