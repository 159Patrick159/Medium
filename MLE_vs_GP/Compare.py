################### HEADER ###################
# Author: Patrick Sandoval                   #
# Date: 2023-12-02                           #
# ############################################
# The following script 'fits' randomly       #
# generated data using a Gaussian process    #
# and a Markov-Chain Monte-Carlo MLE method  #
##############################################

######################### Gaussian Process #########################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman1
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from scipy.optimize import minimize
from Helpers import plot_gpr_samples

##################### NOTES AND COMMENTS #####################
# Kernel is the covariance function - defines correlation between data points
# RBF stands for Radial Basis Function
# C is the constant kernel (amplitude) both are used to define the kernel for the GP

# Generate synthetic data
def HiddenFunc(X):
    return np.ravel(3*X**2 + 4*X - X**3)
# Set seed for generation
rng = np.random.default_rng(seed=42)
X = np.sort(rng.uniform(0,5,5))[:,np.newaxis]
y = HiddenFunc(X) + rng.normal(0,0.1,X.shape[0])

# Define squared exponential kernel with set lenght scale and bounds given data
kernel = 1.0* RBF(length_scale=0.1,length_scale_bounds=(1e-5,1))

# Create a Gaussian Process Regressor and fit to data
gp = GaussianProcessRegressor(kernel=kernel, random_state=42).fit(X,y)

# Look at R^2 value
print("Coefficient of Determinatino (R^2):",round(gp.score(X,y),5))

# Make predictions on new finely sampled data
X_new = np.linspace(0, 5, 100)[:, np.newaxis]
y_pred, cov = gp.predict(X_new, return_cov=True)
# Compute std from covariance matrix
sigma = np.sqrt(np.diag(cov))

# Plot covariance matrix
plt.figure(figsize=(6,6))
plt.title("Covariance Matrix of Joint Predictive Distribution")
plt.imshow(cov)
plt.colorbar()
plt.tight_layout()
plt.savefig("MLE_vs_GP/Figures/CovMatrix.png")

# Plot the results
fig, ax = plt.subplots(figsize=(8,5))
plot_gpr_samples(gpr_model=gp, n_samples=4, ax=ax)
ax.scatter(X, y, c='r', zorder=10, label='Observations')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('Gaussian Process Regression')
ax.legend()
plt.savefig("MLE_vs_GP/Figures/GP.png")
plt.show()
