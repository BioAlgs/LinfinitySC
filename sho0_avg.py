import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from utils.synth import *
from utils.competitors import *

std = False
n_folds = 5
methods = [
    'Synthetic Control',
    'Lasso', 
    'Ridge', 
    'L-inf', 
    'L1 + L-inf',
    'Elastic net', 
]

sectors = [
    # 'Basic_Materials', 
    # 'Communication_Services', 
    # 'Consumer_Cyclical', 
    'Consumer_Defensive', 
    # 'Energy', 
    # 'Healthcare', 
    # 'Industrials',
    # 'Technology'
]

# Define constants
START_TIME = '2005-01-01'
INTERVENTION_TIME = '2005-05-02'
STOP_TIME = '2005-05-31'

START_TIME = datetime.strptime(START_TIME, '%Y-%m-%d')
INTERVENTION_TIME = datetime.strptime(INTERVENTION_TIME, '%Y-%m-%d')
STOP_TIME = datetime.strptime(STOP_TIME, '%Y-%m-%d')


for sector in sectors:
    # Load data specific to the sector
    df = pd.read_csv(f'SHO/{sector}_data.csv')  # Adjusted to load sector-specific data
    df['date'] = pd.to_datetime(df['Date'])
    sho_sector_df = pd.read_csv('SHO/sho_sector_info.csv')
    sector_df = sho_sector_df[sho_sector_df['sector'] == sector.replace('_', ' ')]
    sho0 = sector_df[sector_df['sho_pilot'] == 0]['tsymbol'].tolist()

    print(f"Sector: {sector}, Number of companies: {len(sho0)}")

    # Initialize a list to store relative treatment effects for each method
    relative_ates = []

    for i in range(len(sho0)):
        treated_company = sho0[i]
        # Prepare data for the treated company
        control_companies = [company for company in sho0 if company != treated_company]
        Y_pre = df[df['date'] < INTERVENTION_TIME].drop(columns=['date'])
        Y_post = df[df['date'] >= INTERVENTION_TIME].drop(columns=['date'])
        
        Y1_pre = Y_pre[[treated_company]].to_numpy().astype('float64')
        Y0_pre = Y_pre[control_companies].to_numpy().astype('float64')
        Y1_post = Y_post[[treated_company]].to_numpy().astype('float64')
        Y0_post = Y_post[control_companies].to_numpy().astype('float64')

        # Parameter selection
        T0, J = Y0_pre.shape
        T1 = Y0_post.shape[0]
        idxs1 = np.arange(T0)
        idxs2 = idxs1.copy()

        alpha_inf, lam_inf = param_selector(Y1_pre[idxs1, 0], Y0_pre[idxs1], method='inf', std=std, n_folds=n_folds)
        alpha_l1, lam_l1 = param_selector(Y1_pre[idxs1, 0], Y0_pre[idxs1], method='l1', std=std, n_folds=n_folds)
        alpha_l2, lam_l2 = param_selector(Y1_pre[idxs1, 0], Y0_pre[idxs1], method='l2', std=std, n_folds=n_folds)
        alpha_l1_inf, lam_l1_inf = param_selector(Y1_pre[idxs1, 0], Y0_pre[idxs1], method='l1-inf', std=std, n_folds=n_folds)
        alpha_l1_l2, lam_l1_l2 = param_selector(Y1_pre[idxs1, 0], Y0_pre[idxs1], method='l1-l2', std=std, n_folds=n_folds)

        # Calculate weights
        w_sc = sc(Y1_pre, Y0_pre)
        w_inf = our(Y1_pre[idxs2], Y0_pre[idxs2], alpha_inf, lam_inf, 'inf', std=std)
        w_l1 = our(Y1_pre[idxs2], Y0_pre[idxs2], alpha_l1, lam_l1, 'l1', std=std)
        w_l2 = our(Y1_pre[idxs2], Y0_pre[idxs2], alpha_l2, lam_l2, 'l2', std=std)
        w_l1_inf = our(Y1_pre[idxs2], Y0_pre[idxs2], alpha_l1_inf, lam_l1_inf, 'l1-inf', std=std)
        w_l1_l2 = our(Y1_pre[idxs2], Y0_pre[idxs2], alpha_l1_l2, lam_l1_l2, 'l1-l2', std=std)

        # Calculate treatment effects
        W = np.array([w_sc, w_l1[1:], w_l2[1:], w_inf[1:], w_l1_inf[1:], w_l1_l2[1:]])
        mu = np.array([0, w_l1[0], w_l2[0], w_inf[0], w_l1_inf[0], w_l1_l2[0]])
        SC_outcomes = np.vstack([Y0_pre, Y0_post]).dot(W.T) + mu.reshape(1, len(methods))
        True_outcomes = np.vstack([Y1_pre, Y1_post]).flatten()
        treatment_effect = True_outcomes[T0:, np.newaxis] - SC_outcomes[T0:, :]
        ate = np.mean(treatment_effect, axis=0)

        # Calculate relative treatment effects
        relative_ate = ate / np.mean(Y1_post)
        relative_ates.append(relative_ate)
        if i % 10 == 0:
            print(f"Processed {i} companies for sector {sector}")
        

    # Average relative treatment effects across all treated companies
    # arate = np.mean(np.abs(relative_treatment_effects), axis=0)
    arate = np.mean(relative_ates, axis=0)
    plt.barh(methods, arate * 100, color='green')
    
    # Add numbers to the top of the bars
    for index, value in enumerate(arate * 100):
        if value >= 0:
            plt.text(value, index, f'{value:.2f}', va='center', ha='left')  # Left align for positive values
        else:
            plt.text(value, index, f'{value:.2f}', va='center', ha='right')  # Right align for negative values

    plt.xlabel('Average Relative Treatment Effect (%)')
    plt.ylabel('Methods')
    plt.grid(axis='x')
    plt.gca().invert_yaxis()  # Reverse the order of the y-axis
    plt.tight_layout()
    plt.savefig(f'SHO/ate/{sector}_arate.png')
    plt.close()