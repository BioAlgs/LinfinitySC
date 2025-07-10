from PIL import Image
from matplotlib.patches import Patch, Circle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams.update({
    "font.family": "serif",
})


# Generate data for Ridge and Lasso regression
x = np.linspace(-2, 2, 100)
y1 = x
lam = 0.5
y_ridge = x / (1 + lam * 2)
y_lasso = np.sign(x) * np.maximum(np.abs(x) - lam, np.zeros(100))

# Create figure with GridSpec for side-by-side plots
fig = plt.figure(figsize=(16, 6))  # Width increased to fit three plots
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.2)  # Three plots with spacing

# Plotting Lasso Regression
ax1 = fig.add_subplot(gs[0], aspect='equal')
ax1.plot(x, y1, 'gray', linewidth=1.5)
ax1.plot(x, y_lasso, 'red', ls='--', linewidth=2.0)
ax1.axhline(0, color='black', linewidth=1.5)
ax1.axvline(0, color='black', linewidth=1.5)
ax1.set_title('Lasso', fontsize=28)
ax1.text(2.6, 0, r'$\widehat{\omega}_{2}^{\textrm{OLS}}$', fontsize=25, ha='left', va='center')
ax1.text(0.1, 2.2, r'$\widehat{\omega}^{\textrm{Lasso}}_{2}$', fontsize=25, ha='left', va='center')
ax1.plot(lam, 0, 'ro', markersize=8)
ax1.annotate(r'$\widehat{\omega}_{2}^{\textrm{OLS}} = \lambda$', xy=(lam + 0.1, -0.1), xytext=(lam + 0.5, -1.0), fontsize=25,
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', lw=1.5))
ax1.axis('off')
ax1.set_xlim([-2.5, 2.5])
ax1.set_ylim([-2.5, 2.5])

# Plotting Ridge Regression
ax2 = fig.add_subplot(gs[1], aspect='equal')
ax2.plot(x, y1, 'gray', linewidth=1.5)
ax2.plot(x, y_ridge, 'red', ls='--', linewidth=2.0)
ax2.axhline(0, color='black', linewidth=1.5)
ax2.axvline(0, color='black', linewidth=1.5)
ax2.set_title('Ridge', fontsize=28)
ax2.text(2.6, 0, r'$\widehat{\omega}_{2}^{\textrm{OLS}}$', fontsize=25, ha='left', va='center')
ax2.text(0.1, 2.2, r'$\widehat{\omega}^{\textrm{Ridge}}_{2}$', fontsize=25, ha='left', va='center')
ax2.axis('off')
ax2.set_xlim([-2.5, 2.5])
ax2.set_ylim([-2.5, 2.5])

# Plotting L-infinity Norm
ax3 = fig.add_subplot(gs[2], aspect='equal')
ax3.plot(x, y1, 'gray', linewidth=1.5)
ax3.axhline(0, color='black', linewidth=1.5)
ax3.axvline(0, color='black', linewidth=1.5)
ax3.set_title(r'$L_\infty$', fontsize=28)
ax3.text(2.6, 0, r'$\widehat{\omega}_{2}^{\textrm{OLS}}$', fontsize=25, ha='left', va='center')
ax3.text(0.1, 2.2, r'$\widehat{\omega}_{2}^{\infty}$', fontsize=25, ha='left', va='center')

# Coordinates for the segments
segments = [
    [(1.5, 1), (2, 1.5)],
    [(1, 0.75), (1.5, 1)],
    [(0.5, 0.5), (1, 0.75)],
    [(0, 0), (0.5, 0.5)]
]

# Plot each segment and its symmetric
for segment in segments:
    x_seg, y_seg = zip(*segment)
    ax3.plot(x_seg, y_seg, 'r--', linewidth=2.0)
    ax3.plot([-p for p in x_seg], [-q for q in y_seg], 'r--', linewidth=2.0)

# Add a vertical dashed line at the midpoint using exact positions
plt.plot([1, 1], [0, 0.75], 'k--')  # Adjust color and style as needed


# Annotations for L-infinity
ax3.plot(-0.5, -0.5, 'ro', markersize=8)
ax3.plot(0.5, 0.5, 'ro', markersize=8)
ax3.plot(1.5, 1.0, 'ro', markersize=8)
ax3.plot(1.0, 0.0, 'ko', markersize=8)
ax3.annotate(r'$a$', xy=(-0.5+0.1, -0.5-0.3), fontsize=25)
ax3.annotate(r'$b$', xy=(0.5+0.1, 0.5-0.3), fontsize=25)
ax3.annotate(r'$c$', xy=(1.5+0.1, 1.0-0.3), fontsize=25)
ax3.annotate(r'$|\widehat{\omega}_3^{\textrm{OLS}}|$', xy=(1-0.2, -0.5), fontsize=25)
ax3.annotate(r'$\big||\widehat{\omega}_{2}^{\textrm{OLS}}| - |\widehat{\omega}_{3}^{\textrm{OLS}}|\big| = \lambda$',
             xy=(0.5 - 0.1, 0.5), xytext=(-3.2, 1.5), fontsize=25,
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', lw=1.5))
ax3.annotate(r'$\big||\widehat{\omega}_{2}^{\textrm{OLS}}| - |\widehat{\omega}_{3}^{\textrm{OLS}}|\big| = \lambda$',
             xy=(1.5 - 0.1, 1.0), xytext=(-3.2, 1.5), fontsize=25,
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', lw=1.5))
ax3.annotate(r'$\big||\widehat{\omega}_{2}^{\textrm{OLS}}| - |\widehat{\omega}_{3}^{\textrm{OLS}}|\big| = \lambda$',
             xy=(-0.5, -0.5 + 0.1), xytext=(-3.2, 1.5), fontsize=25,
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', lw=1.5))

ax3.axis('off')
ax3.set_xlim([-2.5, 2.5])
ax3.set_ylim([-2.5, 2.5])

plt.savefig("ex1.png", dpi=500)

# Define the range for omega_2 and omega_3
w2 = np.linspace(0, 2.0, 200)
w3 = np.linspace(0, 2.0, 200)
W2, W3 = np.meshgrid(w2, w3)

# Plot setup
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim([0, 2.0])
ax.set_ylim([0, 2.0])

# Define the region omega_2 + omega_3 <= 1
region = (W2 + W3 <= 1)
# Highlight the omega_2 + omega_3 <= 1 region using hatching
ax.contourf(W2, W3, region, levels=[0.5, 1], hatches=['///'], colors='none', alpha=0)

# Draw the boundary omega_2 + omega_3 = 1
ax.plot(w2, 1 - w2, 'k-', linewidth=2, label=r'$\omega_2 + \omega_3 = 1$')

# Draw dashed lines for separation
ax.plot(w2, w2 - 1, 'k--', linewidth=1.5)  # omega_2 - omega_3 = 1
ax.plot(w2, w2 + 1, 'k--', linewidth=1.5)  # omega_2 - omega_3 = -1

# Add text labels to indicate regions
ax.text(0.4, 1.8, r'$\omega_2 - \omega_3 < -\lambda$', fontsize=18, ha='center', va='center')
ax.text(1.6, 0.1, r'$\omega_2 - \omega_3 > \lambda$', fontsize=18, ha='center', va='center')
ax.text(1.3, 1.3, r'$|\omega_2 - \omega_3| \leq \lambda$', fontsize=18, ha='center', va='center')

# Add points and arrows
# Upper area
upper_point = (0.2, 1.5)
upper_proj = (0, 1.0)  # Projection to (0, λ)

# Middle area
middle_point = (1.0, 1.0)
middle_proj = (0.5, 0.5)  # Projection onto the L1 ball

# Lower area
lower_point = (1.7, 0.5)
lower_proj = (1.0, 0)  # Projection to (λ, 0)

# Plot original points with red circles
ax.plot(*upper_point, 'ro', markersize=6)
ax.plot(*middle_point, 'ro', markersize=6)
ax.plot(*lower_point, 'ro', markersize=6)

# Plot projections with red dashed circles
for proj in [upper_proj, middle_proj, lower_proj]:
    ax.add_patch(Circle(proj, 0.03, edgecolor='red', facecolor='none', linestyle='dashed', linewidth=1.5))

# Draw arrows from original points to their projections
ax.annotate("", xy=upper_proj, xytext=upper_point, arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=middle_proj, xytext=middle_point, arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=lower_proj, xytext=lower_point, arrowprops=dict(arrowstyle="->", lw=1.5))

# Labels and title
ax.set_xlabel(r'$\omega_2$', fontsize=18)
ax.set_ylabel(r'$\omega_3$', fontsize=18)

# Set tick labels at omega_2 = 1 and omega_3 = 1
ax.set_xticks([1.0])
ax.set_xticklabels([r'$\lambda$'], fontsize=15)
ax.set_yticks([1.0])
ax.set_yticklabels([r'$\lambda$'], fontsize=15)

legend_elements = [Patch(facecolor='none', edgecolor='black', hatch='///', label=r'$\omega_2 + \omega_3 \leq 1$')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=18)

# Show plot
plt.savefig('ex1_pre.png', dpi=500)

# Define functions for L1 + L_inf and L1 + L2 norms
def l1_plus_linf(x, y, alpha):
    return np.abs(x) + np.abs(y) + alpha * np.maximum(np.abs(x), np.abs(y))

def l1_plus_l2(x, y, alpha):
    return np.abs(x) + np.abs(y) + alpha * np.sqrt(x**2 + y**2)

# Common grid for contour plots
alpha = 1.0
x = np.linspace(-1.5, 1.5, 400)
y = np.linspace(-1.5, 1.5, 400)
X, Y = np.meshgrid(x, y)

# Calculate Z values for each norm function
Z_l1_linf = l1_plus_linf(X, Y, alpha)
Z_l1_l2 = l1_plus_l2(X, Y, alpha)

# Create a figure with GridSpec
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.2)


# Case 1: Both norms push solution to sparsity (one parameter = 0)
ax1 = fig.add_subplot(gs[0], aspect='equal')
ax1.contour(X, Y, Z_l1_linf, levels=[1], colors='red', linestyles='--')
ax1.contour(X, Y, Z_l1_l2, levels=[1], colors='blue')

ax1.axhline(0, color='black', lw=0.5)
ax1.axvline(0, color='black', lw=0.5)
ax1.set_xlim(-0.8, 1.3)
ax1.set_ylim(-0.8, 1.3)
ax1.axis('off')

# Define ellipse parameters for the first case
width, height = 1.2, 0.7  # Semi-axes lengths for ellipse
center = (0.2, 0.8)
angle = -20
for scale in np.linspace(1, 0.5, 2):
    scaled_width = width * scale
    scaled_height = height * scale
    ellipse1 = patches.Ellipse(center, width=scaled_width, height=scaled_height, angle=angle, edgecolor='black', facecolor='none')
    ax1.add_patch(ellipse1)

# Create custom legend entries
l2_line = mlines.Line2D([], [], color='blue', linestyle='-', label=r'$\|\omega\|_1 + \|\omega\|_2$ = 1')
linf_line = mlines.Line2D([], [], color='red', linestyle='--', label=r'$\|\omega\|_1 + \|\omega\|_{\infty}$ = 1')
ax1.legend(handles=[l2_line, linf_line], loc='lower right', fontsize=14)

w0 = center
ax1.plot(*w0, 'ro')
ax1.annotate(r'$\widehat{\omega}^{\textrm{OLS}}$', w0, textcoords="offset points", xytext=(15, -20), ha='center', fontsize=20)
what = (0, 0.5)
ax1.plot(*what, 'ro')
ax1.annotate(r'$\widehat{\omega}^{\textrm{en}}=\widehat{\omega}^{1,\infty}$', what, textcoords="offset points", xytext=(52, -2), ha='center', fontsize=20)

# Case 2: Different behaviors
ax2 = fig.add_subplot(gs[1], aspect='equal')
ax2.contour(X, Y, Z_l1_linf, levels=[1], colors='red', linestyles='--')
ax2.contour(X, Y, Z_l1_l2, levels=[1], colors='blue')

ax2.axhline(0, color='black', lw=0.5)
ax2.axvline(0, color='black', lw=0.5)
ax2.set_xlim(-0.8, 1.3)
ax2.set_ylim(-0.8, 1.3)
ax2.axis('off')

# Define ellipse parameters for the second case
width, height = 1.2, 0.7
center = (0.49, 0.6)
angle = -30
for scale in [1.0, 0.885, 0.5]:
    scaled_width = width * scale
    scaled_height = height * scale
    ellipse1 = patches.Ellipse(center, width=scaled_width, height=scaled_height, angle=angle, edgecolor='black', facecolor='none')
    ax2.add_patch(ellipse1)

ax2.legend(handles=[l2_line, linf_line], loc='lower right', fontsize=14)

w0 = center
ax2.plot(*w0, 'ro')
ax2.annotate(r'$\widehat{\omega}^{\textrm{OLS}}$', w0, textcoords="offset points", xytext=(15, -20), ha='center', fontsize=20)

what1 = (0.208, 0.368)
ax2.plot(*what1, 'ro')
ax2.annotate(r'$\widehat{\omega}^{\textrm{en}}$', what1, textcoords="offset points", xytext=(-10, -20), ha='center', fontsize=20)

what2 = (0.333, 0.333)
ax2.plot(*what2, 'ro')
ax2.annotate(r'$\widehat{\omega}^{1,\infty}$', what2, textcoords="offset points", xytext=(23, -3), ha='center', fontsize=20)

# plt.tight_layout()
plt.savefig('ex2.png', dpi=500)
