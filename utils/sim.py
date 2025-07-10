import pickle
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from .synth import *
from .competitors import *
from .tools import pollute, generate_ARMA_series

methods = [ 
    'SC',
    'Lasso', 
    'Ridge',
    'Elastic Net',
    'L-inf',
    'L1 + L-inf'
] 

param_methods = [
    'l1', 
    'l2', 
    'l1-l2',
    'inf', 
    'l1-inf'
]  # Methods for param_selector


def sim(stationary, T0, T1, J, sig_F, sig0, sig1, rho1, rho2, gamma, 
        DGP, trt_effect, fixed_alpha, fixed_lam, pollution, 
        std, intercept, n_folds=5, test_size=None):
    
    T01 = T0 + T1
    F1 = np.random.normal(loc=0.0, scale=sig_F, size=T01)
    F2 = np.random.normal(loc=0.0, scale=sig_F, size=T01)

    lambda1 = np.arange(1, J + 1) / J
    lambda2 = lambda1.copy()
    
    if not stationary:
        F2 += np.arange(1, T01 + 1) / T01
        lambda2 = np.square(np.arange(1, J + 1))

    q2, r2 = divmod(J, 2)
    q3, r3 = divmod(J, 3)
    w1 = np.ones(J) / J
    w2 = np.random.uniform(-3/J, 3/J, J)
    w_beta = np.random.beta(0.2, 0.2, J)
    w3 = (w_beta - 0.5) * 3/J
    w4 = np.concatenate((w3[:q2] * 2, [0]*(q2+r2)))
    np.random.shuffle(w4)
    w5 = np.concatenate((w3[:q3] * 3, [0]*(2*q3+r3)))
    # w4 = np.concatenate((np.random.normal(0, 3/J, q2), [0]*(q2+r2)))
    np.random.shuffle(w5)
    w_true = \
        np.outer(w1, (DGP == 1)) + np.outer(w2, (DGP == 2)) + \
        np.outer(w3, (DGP == 3)) + np.outer(w4, (DGP == 4)) + \
        np.outer(w5, (DGP == 5))
    y = np.kron(np.ones(T01), lambda1) + np.kron(F1, np.ones(J)) + \
        np.kron(F2, lambda2)
    # noise1 = np.random.normal(loc=10.0, scale=sig_F, size=(T01, J))  # New noise component
    # noise2 = np.random.normal(loc=10.0, scale=sig_F, size=(T01, J))  # Another noise component
    # y = noise1 + noise2

    epsl = np.empty((T01, J))
    for j in range(J):
        epsl[:, j] = generate_ARMA_series(T01, rho1, gamma)

    Y0 = y.reshape(T01, J) + sig0 * epsl
    u = generate_ARMA_series(T01, rho2, gamma)

    Y1 = Y0 @ w_true + sig1 * u.reshape(-1, 1)
    Y1 = Y1.flatten()
    Y1[T0:] += trt_effect

    ## separate into pre-treatment and post-treatment
    Y0_pre = Y0[:T0,:]
    Y0_post = Y0[T0:,:]
    Y1_pre = Y1[:T0]
    Y1_post = Y1[T0:]

    ## add some contamination to the data
    if pollution is not None:
        Y0_post = pollute(Y0_post, w_true, pollution)

    ## competitors
    # w_did = np.nan
    w_sc = sc(Y1_pre, Y0_pre)    

    ## parameter selector
    # idxs = np.random.choice(T0, T0, replace=False)
    # idxs1, idxs2 = idxs[:T0//2], idxs[T0//2:]
    idxs1 = np.arange(T0)
    idxs2 = idxs1.copy()
    alphas = []
    lambdas = []
    weights = []

    for method in param_methods:
        alpha, lam = param_selector(
            Y1_pre[idxs1], Y0_pre[idxs1], method=method, 
            fixed_alpha=fixed_alpha, fixed_lam=fixed_lam, 
            std=std, intercept=intercept,
            n_folds=n_folds, test_size=test_size)
        alphas.append(alpha)
        lambdas.append(lam)

    for i, method in enumerate(param_methods):
        w = our(
            Y1_pre[idxs2], Y0_pre[idxs2], alphas[i], lambdas[i], method, 
            std=std, intercept=intercept)
        weights.append(w)

    # tau_did = did(Y1, Y0)[T0:]
    if intercept:
        Y0_post_plus = np.hstack([np.ones((Y1_post.shape[0], 1)), Y0_post])
    else:
        Y0_post_plus = Y0_post

    tau_oracle = Y1_post - Y0_post @ w_true.flatten()
    tau_sc = Y1_post - Y0_post @ w_sc
    taus = [tau_sc]

    for w in weights:
        tau = Y1_post - Y0_post_plus @ w
        taus.append(tau)

    # Initialize lists to store metrics
    # weight metrics
    if intercept:
        W = [w_sc] + [w[1:] for w in weights]
    else:
        W = [w_sc] + weights

    # tau metrics
    Tau = [tau_oracle] + taus 
    # alpha and lambda metrics
    Alpha = [alphas[4], alphas[2]]  # l1-inf and l1-l2
    Lambda = lambdas

    return (w_true.flatten(), W, Tau, Alpha, Lambda)


# Function to run simulations for given parameters and aggregate results
def sim_wrapper(filepath, stationary, T0, T1, J, sig_F, sig_Y0, sig_Y1, 
                rho1, rho2, gamma, DGP, trt_effect, fixed_alpha, fixed_lam, 
                pollution=None, std=False, intercept=True, n_rep=1000):
    threshold = 1e-4
    # Initialize accumulators for each metric
    combined_dens = np.zeros((len(methods), n_rep))
    combined_mag = combined_dens.copy()
    combined_bias_w = combined_dens.copy()
    combined_mse_w = combined_dens.copy()
    combined_tau = np.zeros((len(methods) + 1, n_rep, T1))
    combined_alpha = np.zeros((2, n_rep))
    combined_lam = np.zeros((len(methods) - 1, n_rep))

    # Parallelize the repetitions
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(sim, stationary, T0, T1, J, sig_F, sig_Y0, sig_Y1, 
                                   rho1, rho2, gamma, DGP, trt_effect, fixed_alpha, fixed_lam, 
                                   pollution, intercept, std) for r in range(n_rep)]
        results = [future.result() for future in futures]

    # Save the results using pickle
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

    for r, result in enumerate(results):
        w_true, W, Tau, Alpha, Lambda = result
        dens, mag, bias_w, mse_w = [], [], [], []

        for i in range(len(W)):
            # Density
            dens.append(np.count_nonzero(W[i] > threshold))
            # Magnitude
            mag.append(np.max(np.abs(W[i])))
            # bias for w
            bias_w.append(np.mean(W[i] - w_true))
            # RMSE for w
            mse_w.append(np.mean(np.square(W[i] - w_true)))

            combined_tau[i, r, :] = Tau[i]

        # oracle tau
        combined_tau[i + 1, r, :] = Tau[i + 1]

        combined_dens[:, r] = dens
        combined_mag[:, r] = mag
        combined_bias_w[:, r] = bias_w
        combined_mse_w[:, r] = mse_w
        combined_alpha[:, r] = Alpha
        combined_lam[:, r] = Lambda

    w_mat = np.column_stack((np.abs(np.mean(combined_dens, axis=1)),
                             np.abs(np.mean(combined_mag, axis=1)),
                             np.abs(np.mean(combined_bias_w, axis=1)),
                             np.sqrt(np.mean(combined_mse_w, axis=1))))
    
    # Calculate bias and RMSE for each post-treatment period
    bias = np.abs(np.mean(combined_tau - trt_effect, axis=1))
    rmse = np.sqrt(np.mean(np.square(combined_tau - trt_effect), axis=1))

    # Create a vector of trt_effect
    trt_effect_vector = np.full(T1, trt_effect)
    
    # Calculate cumulative ATE bias and RMSE for each epoch
    ate_true = np.cumsum(trt_effect_vector) / np.arange(1, T1 + 1)
    ate_est = np.cumsum(combined_tau, axis=2) / np.arange(1, T1 + 1)
    ate_bias = np.abs(np.mean(ate_est - ate_true, axis=1))
    ate_rmse = np.sqrt(np.mean(np.square(ate_est - ate_true), axis=1))

    return w_mat, bias, rmse, ate_bias, ate_rmse, combined_alpha, combined_lam

def calculate_taus(response_df, shock_values, INTERVENTION_TIME, START_TIME, STOP_TIME, unit_name):
    """
    Calculates taus for various methods given a response DataFrame and shock values.

    Args:
        response_df (pd.DataFrame): The original response DataFrame.
        shock_values (np.ndarray): Shock values to apply to the Utah column.
        INTERVENTION_TIME (int): Time of intervention.
        START_TIME (int): Start time of the data.
        STOP_TIME (int): Stop time of the data.
        index (int): Index of the state to extract weights.

    Returns:
        list: A list of taus for each method across the shock values.
    """
    response_df_shocked = response_df.copy()
    response_df_shocked.loc[1989:2001, unit_name] += shock_values.astype('float32')
    california_response_df_shocked = response_df_shocked[['California']]
    non_california_response_df_shocked = response_df_shocked.drop(columns='California')

    california_pre_1988_df_shocked = california_response_df_shocked[california_response_df_shocked.index <= 1988]
    california_post_1988_df_shocked = california_response_df_shocked[california_response_df_shocked.index > 1988]
    non_california_pre_1988_df_shocked = non_california_response_df_shocked[non_california_response_df_shocked.index <= 1988]
    non_california_post_1988_df_shocked = non_california_response_df_shocked[non_california_response_df_shocked.index > 1988]

    Z0 = non_california_pre_1988_df_shocked
    Z1 = california_pre_1988_df_shocked
    Y0 = non_california_post_1988_df_shocked
    Y1 = california_post_1988_df_shocked

    T0 = INTERVENTION_TIME - START_TIME
    T1 = STOP_TIME - INTERVENTION_TIME
    test_size = 0.4

    Y1_pre = Z1.to_numpy().astype('float64')
    Y0_pre = Z0.to_numpy().astype('float64')
    Y1_post = Y1.to_numpy().astype('float64')
    Y0_post = Y0.to_numpy().astype('float64')

    ## competitors
    # w_did = np.nan
    w_sc = sc(Y1_pre, Y0_pre)    

    ## parameter selector
    alphas = []
    lambdas = []
    weights = []

    for method in param_methods:
        alpha, lam = param_selector(
            Y1_pre, Y0_pre, method=method, test_size=test_size)
        alphas.append(alpha)
        lambdas.append(lam)

    for i, method in enumerate(param_methods):
        w = our(
            Y1_pre, Y0_pre, alphas[i], lambdas[i], method)
        weights.append(w)

    Y0_post_plus = np.hstack([np.ones((Y1_post.shape[0], 1)), Y0_post])
    
    tau_sc = Y1_post[:,0] - Y0_post @ w_sc
    taus = [tau_sc]

    for w in weights:
        tau = Y1_post[:,0] - Y0_post_plus @ w
        taus.append(tau)
        
    taus = np.array(taus)

    return taus
