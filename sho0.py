import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from utils.synth import *
from utils.competitors import *


std = False
n_folds = None
methods = [
    'Synthetic Control',
    'Lasso', 
    'Ridge', 
    'L-inf', 
    'L1 + L-inf',
    'Elastic net', 
]

# Define constants
START_TIME = '2005-01-01'
INTERVENTION_TIME = '2005-05-02'
STOP_TIME = '2005-05-31'

START_TIME = datetime.strptime(START_TIME, '%Y-%m-%d')
INTERVENTION_TIME = datetime.strptime(INTERVENTION_TIME, '%Y-%m-%d')
STOP_TIME = datetime.strptime(STOP_TIME, '%Y-%m-%d')

# Load data
df = pd.read_csv('SHO/Technology_data.csv')
df['date'] = pd.to_datetime(df['Date'])
sho_sector_df = pd.read_csv('SHO/sho_sector_info.csv')
tech_sector_df = sho_sector_df[sho_sector_df['sector'] == 'Technology']
sho0 = tech_sector_df[tech_sector_df['sho_pilot'] == 0]['tsymbol'].tolist()

# List of companies to analyze
companies_to_analyze = sho0[:5]  # Add more companies as needed

for treated_company in companies_to_analyze:
    # Prepare data for the treated company
    control_companies = [company for company in sho0 if company != treated_company]
    Y_pre = df[df['date'] < INTERVENTION_TIME].drop(columns=['date'])
    Y_post = df[df['date'] >= INTERVENTION_TIME].drop(columns=['date'])
    
    Y1_pre = Y_pre[[treated_company]].to_numpy().astype('float64')
    Y0_pre = Y_pre[control_companies].to_numpy().astype('float64')
    Y1_post = Y_post[[treated_company]].to_numpy().astype('float64')
    Y0_post = Y_post[control_companies].to_numpy().astype('float64')

    # Parameter selection
    T0, n_folds = Y0_pre.shape
    T1 = Y0_post.shape[0]

    alpha_inf, lam_inf = param_selector(Y1_pre[:, 0], Y0_pre, method='inf', std=std, test_size=0.4)
    alpha_l1, lam_l1 = param_selector(Y1_pre[:, 0], Y0_pre, method='l1', std=std, test_size=0.4)
    alpha_l2, lam_l2 = param_selector(Y1_pre[:, 0], Y0_pre, method='l2', std=std, test_size=0.4)
    alpha_l1_inf, lam_l1_inf = param_selector(Y1_pre[:, 0], Y0_pre, method='l1-inf', std=std, test_size=0.4)
    alpha_l1_l2, lam_l1_l2 = param_selector(Y1_pre[:, 0], Y0_pre, method='l1-l2', std=std, test_size=0.4)

    # Calculate weights
    w_sc = sc(Y1_pre, Y0_pre)
    w_inf = our(Y1_pre, Y0_pre, alpha_inf, lam_inf, 'inf', std=std)
    w_l1 = our(Y1_pre, Y0_pre, alpha_l1, lam_l1, 'l1', std=std)
    w_l2 = our(Y1_pre, Y0_pre, alpha_l2, lam_l2, 'l2', std=std)
    w_l1_inf = our(Y1_pre, Y0_pre, alpha_l1_inf, lam_l1_inf, 'l1-inf', std=std)
    w_l1_l2 = our(Y1_pre, Y0_pre, alpha_l1_l2, lam_l1_l2, 'l1-l2', std=std)

    # Calculate treatment effects
    W = np.array([w_sc, w_l1[1:], w_l2[1:], w_inf[1:], w_l1_inf[1:], w_l1_l2[1:]])
    mu = np.array([0, w_l1[0], w_l2[0], w_inf[0], w_l1_inf[0], w_l1_l2[0]])
    SC_outcomes = np.vstack([Y0_pre, Y0_post]).dot(W.T) + mu.reshape(1, len(methods))
    True_outcomes = np.vstack([Y1_pre, Y1_post]).flatten()
    # Combine treatment effects
    treatment_effect = True_outcomes[T0:, np.newaxis] - SC_outcomes[T0:, :]
    
    # Plotting average treatment effects
    average_treatment_effects = np.mean(treatment_effect, axis=0)
    
    fig = plt.figure(figsize=(8, 5))
    plt.barh(methods, average_treatment_effects, color='blue')
    plt.xlabel('Average Treatment Effect')
    plt.ylabel('Methods')
    plt.grid(axis='x')
    plt.gca().invert_yaxis()  # Reverse the order of the y-axis
    plt.tight_layout()
    plt.savefig(f'SHO/ate/{treated_company}.png')
    plt.close()

    print(treated_company)