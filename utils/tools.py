import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib import colors
from scipy.stats import t
from matplotlib.backends.backend_pdf import PdfPages

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams.update({
    "font.family": "serif",
})


def pollute(Y0, w, pollution=3, df=2):
    """
    Add random noise from a t-distribution to proportion p of the columns of the given matrix.

    Args:
    Y0 (numpy.ndarray): The original matrix.
    df (int): Degrees of freedom for the t-distribution.

    Returns:
    numpy.ndarray: The contaminated matrix.
    """
    T, J = Y0.shape
    nonzero_indices = np.nonzero(w)[0]
    # Contaminate #[pollution] of columns
    num_units = min(len(nonzero_indices), pollution)
    # Select random columns
    # selected_units = np.random.choice(nonzero_indices, 
    #                                   num_units, 
    #                                   replace=False)
    selected_units = nonzero_indices[np.argsort(-np.abs(Y0[:, nonzero_indices]).max(axis=0))[:num_units]]
    
    # # Generate random noise using t-distribution
    # noise = t.rvs(df, size=(T, num_units))
    # for i, j in enumerate(selected_units):
    #     Y0[:, j] += noise[:, i]

    noise = t.rvs(df, size=num_units)  # Scalar noise for each column
    for i, col in enumerate(selected_units):
        Y0[:, col] += noise[i]

    return Y0


def generate_AR_series(T01, rho):
    epsl = np.random.normal(size=T01) * np.sqrt(1 - rho**2)
    startvalue = np.random.normal()
    u = np.empty(T01)
    u[0] = rho * startvalue + epsl[0]
    for t in range(1, T01):
        u[t] = rho * u[t-1] + epsl[t]
    return u

def generate_ARMA_series(T01, rho, gamma):
    # White noise terms for AR and MA components
    epsl = np.random.normal(size=T01) * np.sqrt(1 - rho**2 - gamma**2)
    eta = np.random.normal(size=T01)  # Additional noise for MA part
    # Start values for the series
    start_value = np.random.normal()
    u = np.empty(T01)
    # Initial value computation with AR and MA components
    u[0] = rho * start_value + epsl[0] + gamma * eta[0]
    for t in range(1, T01):
        u[t] = rho * u[t - 1] + epsl[t] + gamma * eta[t - 1]  # MA component uses eta[t-1]
    return u


methods = [ 
    'SC',
    'Lasso', 
    'Ridge',
    'Elastic Net',
    r'$L_{\infty}$',
    r'$L_1 + L_{\infty}$'
] 
methods2 = methods.copy()
methods2 = ['Oracle'] + methods


colors = ['b', 'g', 'r', 'c', 'm', 'y', 'grey', 'purple', 'brown', 'pink']
markers = ['o', 's', '^', 'X', '*', '+', 'D', 'p', 'v']

def generate_plots_and_tables(w_mat, bias, rmse, ate_bias, ate_rmse, combined_alpha, combined_K, plot_name):
    """
    Generates a series of plots and tables and saves them into a PDF.

    :param w_mat: A numpy array with metrics for weights.
    :param bias: A numpy array with bias data.
    :param rmse: A numpy array with rmse data.
    :param ate_bias: A numpy array with cumulative ATE bias data.
    :param ate_rmse: A numpy array with cumulative ATE RMSE data.
    :param plot_name: Name of the PDF file to save the plots.
    """
    T1 = rmse.shape[1]
    with PdfPages(plot_name) as pdf:
        
        # ## 1st plot 
        # plt.figure(figsize=(7, 6))
        # plt.barh(range(len(methods)), data[::-1,0])
        # plt.yticks(range(len(methods)), methods[::-1])
        # plt.ylabel('Method')
        # plt.xlabel('Density')
        # plt.tight_layout()
        # pdf.savefig()
        # plt.close()
        # ## 2nd plot 
        # plt.figure(figsize=(7, 6))
        # plt.barh(range(len(methods)), data[::-1,1])
        # plt.yticks(range(len(methods)), methods[::-1])
        # plt.ylabel('Method')
        # plt.xlabel('Magnitude')
        # plt.tight_layout()
        # pdf.savefig()
        # plt.close()
        # ## 3rd plot
        # plt.figure(figsize=(6, 6))
        # for i in range(len(methods)):
        #     plt.plot(range(1, T1 + 1), bias[i, :], color=colors[i % len(colors)], label=methods[i], marker=markers[i])
        # plt.ylim(0, np.max(bias) * 1.1)
        # plt.xlabel("Post Treatment Period")
        # plt.ylabel("Bias")
        # plt.legend(bbox_to_anchor=(1,1), fancybox=True, framealpha=0.5, fontsize='small')
        # pdf.savefig(bbox_inches='tight')
        # plt.close()
        # ## 4th plot
        # plt.figure(figsize=(6, 6))
        # for i in range(len(methods)):
        #     plt.plot(range(1, T1 + 1), rmse[i, :], color=colors[i % len(colors)], label=methods[i], marker=markers[i])
        # plt.ylim(0, np.max(rmse) * 1.1)
        # plt.xlabel("Post Treatment Period")
        # plt.ylabel("RMSE")
        # plt.legend(bbox_to_anchor=(1,1), fancybox=True, framealpha=0.5, fontsize='small')
        # pdf.savefig(bbox_inches='tight')
        # plt.close()

        ## attach the table as well

        # Indices for selecting every ?th epoch
        selected_indices = [i for i in range(T1) if i % 3 == 0]

        metrics = {
            'Cumulative RMSE': np.cumsum(rmse, axis=1),
            'Average TE RMSE': ate_rmse,
            'Bias': bias,
            'Average TE Bias': ate_bias
        }

        for title, data in metrics.items():
            df = pd.DataFrame(data, index=methods2, columns=[f'$T_0$ + {i+1}' for i in range(data.shape[1])])
            df = df.iloc[:, selected_indices]  # Subsetting the DataFrame
            df = df.map(lambda x: np.round(x, 4))
            df_display = df.map(lambda x: f"{x:.4f}")  # Format numbers to 4 decimal places as strings

            fig, ax = plt.subplots(figsize=(len(selected_indices)+1, 2))  
            ax.axis('off')
            table = ax.table(cellText=df_display.values, colLabels=df_display.columns, 
                             rowLabels=df_display.index, loc='center', cellLoc='center')

            for i, column in enumerate(df.columns):
                values = df[column][1:].sort_values()  # Sort values, excluding the first row
                min_val = values.min()
                second_min_val = values[values > min_val].min() if len(values[values > min_val]) > 0 else None
                for j, row_val in enumerate(df[column][1:]):  # Iterate only up to the second-to-last row
                    if row_val == min_val:
                        # +2 to skip header and Oracle
                        table[(j+2, i)].set_facecolor('#ffff99')  # Light yellow for smallest
                    elif row_val == second_min_val:
                        table[(j+2, i)].set_facecolor('#90EE90')  # Light green for second smallest
            plt.suptitle(title, fontsize=14)
            pdf.savefig(bbox_inches='tight')
            plt.close(fig)

        ## weight metrics
        df = pd.DataFrame(w_mat, index=methods, columns=['Density', 'Magnitude', 'Bias', 'RMSE'])
        df = df.map(lambda x: np.round(x, 4))
        # Create a formatted version for display
        df_display = df.map(lambda x: f"{x:.4f}")  # Format numbers to 4 decimal places as strings
        fig, ax = plt.subplots(figsize=(5, 2))  
        ax.axis('off')
        table = ax.table(cellText=df_display.values, colLabels=df_display.columns, 
                         rowLabels=df_display.index, loc='center', cellLoc='center')
        for i, column in enumerate(df.columns[2:]):
            values = df[column].sort_values()  # Sort values
            min_val = values.min()
            second_min_val = values[values > min_val].min() if len(values[values > min_val]) > 0 else None
            for j, row_val in enumerate(df[column]):
                if row_val == min_val:
                    # +1 to skip header 
                    table[(j+1, i+2)].set_facecolor('#ffff99')  # Light yellow for smallest
                elif row_val == second_min_val:
                    table[(j+1, i+2)].set_facecolor('#90EE90')  # Light green for second smallest
        plt.suptitle('Metrics for the Weights', fontsize=14)
        pdf.savefig(bbox_inches='tight')
        plt.close(fig)

        ## Alphas
        plt.figure(figsize=(6, 5))
        _ = plt.hist(combined_alpha[0,:], bins=25, range=(0, 1))
        plt.title(r'$\alpha$ for $L_1 + L_{\infty}$')

        # Calculate statistics
        mean_alpha = np.mean(combined_alpha[0, :])
        median_alpha = np.median(combined_alpha[0, :])
        q1_alpha = np.percentile(combined_alpha[0, :], 25)
        q3_alpha = np.percentile(combined_alpha[0, :], 75)

        # Add text annotations for statistics
        plt.text(0.8, 0.9 * plt.ylim()[1], f'Mean: {mean_alpha:.3f}', fontsize=10)
        plt.text(0.8, 0.85 * plt.ylim()[1], f'Median: {median_alpha:.3f}', fontsize=10)
        plt.text(0.8, 0.8 * plt.ylim()[1], f'1st Q: {q1_alpha:.3f}', fontsize=10)
        plt.text(0.8, 0.75 * plt.ylim()[1], f'3rd Q: {q3_alpha:.3f}', fontsize=10)

        pdf.savefig(bbox_inches='tight')
        plt.close()   
        plt.figure(figsize=(6, 5))
        _ = plt.hist(combined_alpha[1,:], bins=25, range=(0, 1))
        plt.title(r'$\alpha$ for Elastic Net')
        pdf.savefig(bbox_inches='tight')
        plt.close()   




        ## Lambdas
        combined_K = np.log(combined_K)
        plt.figure(figsize=(6, 5))
        _ = plt.hist(combined_K[0,:], bins=20)
        plt.title(r'$\log(\lambda)$ for Lasso')
        pdf.savefig(bbox_inches='tight')
        plt.close()   
        plt.figure(figsize=(6, 5))
        _ = plt.hist(combined_K[1,:], bins=20)
        plt.title(r'$\log(\lambda)$ for Ridge')
        pdf.savefig(bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(6, 5))
        _ = plt.hist(combined_K[2,:], bins=20)
        plt.title(r'$\log(\lambda)$ for $L_{\infty}$')
        pdf.savefig(bbox_inches='tight')
        plt.close()   
        plt.figure(figsize=(6, 5))
        _ = plt.hist(combined_K[3,:], bins=20)
        plt.title(r'$\log(\lambda)$ for $L_1 + L_{\infty}$')
        pdf.savefig(bbox_inches='tight')
        plt.close()   
        plt.figure(figsize=(6, 5))
        _ = plt.hist(combined_K[4,:], bins=20)
        plt.title(r'$\log(\lambda)$ for Elastic Net')
        pdf.savefig(bbox_inches='tight')
        plt.close()
