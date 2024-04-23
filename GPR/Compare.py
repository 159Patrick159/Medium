################### HEADER ###################
# Author: Patrick Sandoval                   #
# Date: 2023-12-02                           #
# ############################################
# The following script 'fits' data generated #
# by some unkwon function using a Gaussian   #
# Process Regressor through distinct kernels #
##############################################

######################### Gaussian Process #########################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic as RQ
from Helpers import plot_gpr_samples

##################### NOTES AND COMMENTS #####################
# Kernel is the covariance function - defines correlation between data points
# RBF stands for Radial Basis Function also known as Squared Exponential

# Generate synthetic data
def HiddenFunc(X):
    return np.ravel(3*X**2 + 4*X - X**3)
# Set seed for generation
rng = np.random.default_rng(seed=42)
X = np.sort(rng.uniform(0,5,5))[:,np.newaxis]
y = HiddenFunc(X) + rng.normal(0,0.1,X.shape[0])


# Plot raw data
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(X,y,c='k')
ax.set_xlabel("X",fontsize=14)
ax.set_ylabel("Observations",fontsize=14)
plt.savefig("GPR/Figures/Raw.png")


fig, ax = plt.subplots(figsize=(8,6))
ax.plot(X,y,'-o',c='k',label='Prediction')
ax.set_xlabel("X",fontsize=14)
ax.set_ylabel("Observations",fontsize=14)
ax.legend()
plt.savefig("GPR/Figures/Raw_LinPred.png")

# Generate points at x=1.3, x=1.8, x=2.8
ysample1 = np.random.normal(loc=7.5,scale=4,size=20)
ysample2 = np.random.normal(loc=11,scale=4,size=20)
ysample3 = np.random.normal(loc=10,scale=4,size=20)
xsample = np.ones(len(ysample1))

fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlabel("X",fontsize=14)
ax.set_ylabel("Observations",fontsize=14)
for i in range(len(ysample1)):
    x = np.append(X,[1.3*xsample[i],1.8*xsample[i],2.8*xsample[i]])
    s = np.argsort(x)
    x = np.sort(x)
    ytmp = np.append(y,[ysample1[i],ysample2[i],ysample3[i]])
    ytmp = ytmp[s]
    ax.plot(x,ytmp,'-o',c='blue',alpha=0.3)
ax.scatter(X,y,c='k',zorder=2,label='Observations')
ax.legend()
plt.savefig("GPR/Figures/Raw_Samples.png")


# Define squared exponential kernel with set lenght scale and bounds given data
RBFkernel = 1.0*RBF(length_scale=0.1,length_scale_bounds=(1e-5,1))
RQkernel = 1.0*RQ(length_scale=0.1,length_scale_bounds=(1e-5,1))

# Create a Gaussian Process Regressor for specific kernel
gp_RBF = GaussianProcessRegressor(kernel=RBFkernel, random_state=42)
gp_RQ = GaussianProcessRegressor(kernel=RQkernel, random_state=42)
 
# Plot prior samples from each kernel
fig, ax = plt.subplots(figsize=(8,8),nrows=2,sharex=True,sharey=True)
plot_gpr_samples(gpr_model=gp_RBF,n_samples=5,ax=ax[0])
plot_gpr_samples(gpr_model=gp_RQ,n_samples=5,ax=ax[1])
ax[0].set_title("Radial Basis Function Kernel Prior Samples")
ax[1].set_title("Rational Quadratic Kernel Prior Samples")
plt.savefig("GPR/Figures/PriorSample.png")

# Fit both models to observations
gp_RBF.fit(X,y)
gp_RQ.fit(X,y)

# Print model evaluation and descriptors
print("Radial Basis Funcion Kernel Description:")
print(f"Coefficient of Determinatino (R^2): {gp_RBF.score(X,y)}\n"
      f"Kernel parameters after fit: \n{gp_RBF.kernel_} \n"
      f"Log-likelihood: {gp_RBF.log_marginal_likelihood(gp_RBF.kernel_.theta):.3f}\n")

print("Rational Quadratic Kernel Description:")
print(f"Coefficient of Determinatino (R^2): {gp_RQ.score(X,y)}\n"
      f"Kernel parameters after fit: \n{gp_RQ.kernel_} \n"
      f"Log-likelihood: {gp_RQ.log_marginal_likelihood(gp_RQ.kernel_.theta):.3f}\n")

# Make predictions on new finely sampled data
X_new = np.linspace(0, 5, 100)[:, np.newaxis]
y_predRBF, covRBF = gp_RBF.predict(X_new, return_cov=True)
y_predRQ, covRQ = gp_RQ.predict(X_new,return_cov=True)

# Plot GP of RBF with n=30 samples
fig, ax = plt.subplots(figsize=(8,6))
n_samples=30
x_fine = np.linspace(0, 5, 100)
X_fine = x_fine.reshape(-1, 1)
y_mean, y_std = gp_RBF.predict(X_fine, return_std=True)
y_samples = gp_RBF.sample_y(X_fine, n_samples)

for idx, single_prior in enumerate(y_samples.T):
    ax.plot(
        x_fine,
        single_prior,
        c='blue',
        alpha=0.3
    )
ax.scatter(X, y, c='k', zorder=10, label='Observations')
ax.legend()
ax.set_xlabel("X",fontsize=14)
ax.set_ylabel("Obsevations",fontsize=14)
plt.savefig("GPR/Figures/GP_Samples.png")
plt.show()

# Plot covariance matrix
fig, ax = plt.subplots(figsize=(10,6),ncols=2)
ax[0].set_title("Squared Exponential Kernel",fontsize=14)
ax[1].set_title("Rational Quadratic Kernel",fontsize=14)
ax[0].set_xlabel(r"X$_{i}$",fontsize=14)
ax[0].set_ylabel(r"X$_{j}$",fontsize=14)
ax[1].set_xlabel(r"X$_{i}$",fontsize=14)
ax[1].set_ylabel(r"X$_{j}$",fontsize=14)
im0 = ax[0].imshow(covRBF)
im1 = ax[1].imshow(covRQ)
cbar0 = plt.colorbar(
    im0, ax=ax[0], fraction=0.045, pad=0.05)
cbar1 = plt.colorbar(
    im1, ax=ax[1], fraction=0.045, pad=0.05)
cbar0.ax.set_ylabel('$k(x,x)$', fontsize=10)
cbar1.ax.set_ylabel('$k(x,x)$', fontsize=10)
ax[0].grid(False)
ax[1].grid(False)
plt.tight_layout()
plt.savefig("GPR/Figures/CovMatrix.png")

# Plot the results
fig, ax = plt.subplots(figsize=(8,10),nrows=2)
#fig.suptitle("Samples from Posterior Distribution")
plot_gpr_samples(gpr_model=gp_RBF, n_samples=4, ax=ax[0])
plot_gpr_samples(gpr_model=gp_RQ, n_samples=4, ax=ax[1])
ax[0].scatter(X, y, c='r', zorder=10, label='Observations')
ax[1].scatter(X, y, c='r', zorder=10, label='Observations')
ax[0].set_title("Squared-Exponential Kernel")
ax[1].set_title("Rational Qudratic Kernel")
plt.savefig("GPR/Figures/GP.png")
plt.show()
