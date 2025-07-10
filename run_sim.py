# Import necessary libraries
import os
import sys
import time
import datetime
import inspect
from utils.sim import *
from utils.tools import generate_plots_and_tables
from itertools import product

# Initial setup
T1 = 11
stationary = 1
trt_effect = 3
sig_F = np.sqrt(2.0)
n_rep = 2000
fixed_alpha = None
fixed_lam = None
pollution = None

# Parameter ranges
T0_J_combinations = [(100, 30)]
rho_gamma_combinations = [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1)]
sig0_values = [np.sqrt(2.0)]
sig1_values = [np.sqrt(1.0)]
dgp_values = [1, 2, 3, 4, 5]
std_values = [False]  # values for std
intercept_values = [True]  # values for intercept

# Generate all parameter combinations
param_combinations = product(
    T0_J_combinations, std_values, intercept_values, 
    sig0_values, sig1_values, rho_gamma_combinations, dgp_values
)

# Get current timestamp
now = datetime.datetime.now()
timestamp = now.strftime("%Y_%m%d_%H%M")

## create folders
os.makedirs(f'sim_results/{timestamp}/data', exist_ok=True)
os.makedirs(f'sim_results/{timestamp}/graphs', exist_ok=True)

# Redirect stdout globally
output_file = open(f"sim_results/{timestamp}/params.log", "w")
sys.stdout = output_file

# Main code to run all parameter combinations
if __name__ == '__main__':

    # Record the setting of this run
    params = {
        "Fixed": {
            "T1": T1, "stationary": stationary, "trt_effect": trt_effect, 
            "sig_F": sig_F, "n_rep": n_rep, 
            "fixed_alpha": fixed_alpha, "fixed_lam": fixed_lam,
            "pollution": pollution
        },
        "Ranges": {
            "T0_J": T0_J_combinations, "rho_gamma": rho_gamma_combinations,
            "sig0": sig0_values, "sig1": sig1_values, "DGP": dgp_values, 
            "std": std_values, "intercept": intercept_values
        }
    }
    print("Simulation Parameters:")
    for category, values in params.items():
        print(f"\n{category}:")
        for key, value in values.items():
            print(f"  {key}: {value}")

    print("\nStarting simulations...\n")

    # Loop through all parameter combinations
    for (T0, J), std, intercept, sig0, sig1, (rho, gamma), dgp in param_combinations:
        start_time = time.time()  # Start time

        # Generate a unique filename based on the simulation parameters
        filepath = f'sim_results/{timestamp}/data/T0_{T0}_J_{J}_std_{int(std)}_int_{int(intercept)}_sig0_{sig0:.2g}_sig1_{sig1:.2g}_rho_{rho:.2g}_gamma_{gamma:.2g}_dgp_{dgp}.pkl'

        w_mat, bias, rmse, ate_bias, ate_rmse, combined_alpha, combined_lam = sim_wrapper(
            filepath=filepath, stationary=stationary, T0=T0, T1=T1, J=J, 
            sig_F=sig_F, sig_Y0=sig0, sig_Y1=sig1,
            rho1=rho, rho2=rho, gamma=gamma, 
            DGP=dgp, trt_effect=trt_effect,
            fixed_alpha=fixed_alpha, fixed_lam=fixed_lam,
            std=std, intercept=intercept,
            pollution=pollution, n_rep=n_rep)
                        
        plot_name = f'sim_results/{timestamp}/graphs/T0_{T0}_J_{J}_std_{int(std)}_int_{int(intercept)}_sig0_{sig0:.2g}_sig1_{sig1:.2g}_rho_{rho:.2g}_gamma_{gamma:.2g}_dgp_{dgp}.pdf'  
        generate_plots_and_tables(w_mat, bias, rmse, ate_bias, ate_rmse, 
                                  combined_alpha, combined_lam, plot_name)

        end_time = time.time()  # End time
        elapsed_time = end_time - start_time
        formatted_time = str(datetime.timedelta(seconds=elapsed_time))
        print(f'DGP {dgp} with std={int(std)}, intercept={int(intercept)}, sig0={sig0:.2g}, sig1={sig1:.2g}, rho1=rho2={rho:.2g}, gamma={gamma:.2g}, took {formatted_time} to run (h:mm:ss).')
    print('\nSimulations finished.\n')
    
    # Print the function body of sim 
    print("\nFunction Body of 'sim':")
    sim_source = inspect.getsource(sim)
    print(sim_source)