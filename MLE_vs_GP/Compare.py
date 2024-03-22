################### HEADER ###################
# Author: Patrick Sandoval                   #
# Date: 2023-12-02                           #
# ############################################
# The following script 'fits' randomly       #
# generated data using a Gaussian process    #
# and a Markov-Chain Monte-Carlo method      #
##############################################

######################### Gaussian Process #########################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, ConstantKernel as C
from scipy.optimize import minimize

##################### NOTES AND COMMENTS #####################
# Kernel is basically the covariance function - defines correlation between data points
# RBF stands for Radial Basis Function
# C is the constant kernel (amplitude) both are used to define the kernel for the GP

# Generate synthetic data with a periodic component
rng = np.random.default_rng(seed=42)
X = np.sort(rng.uniform(0, 5, 20))[:, np.newaxis]
y = np.sin(X).ravel() + rng.normal(0, 0.1, X.shape[0])

# Define the kernel for the Gaussian Process with periodic component
def kernel(periodicity):
    return C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0) * ExpSineSquared(length_scale=1.0, periodicity=periodicity)

# Define the negative log likelihood function for optimization
def neg_log_likelihood(params):
    periodicity = params[0]
    gp = GaussianProcessRegressor(kernel=kernel(periodicity), n_restarts_optimizer=0, random_state=42)
    gp.fit(X, y)
    return -gp.log_marginal_likelihood()

# Initial guess for the periodicity hyperparameter
initial_periodicity = 1.0

# Perform optimization
result = minimize(neg_log_likelihood, [initial_periodicity], bounds=[(1e-3, 1e3)], method='L-BFGS-B')

# Retrieve the optimized periodicity
optimal_periodicity = result.x[0]

# Create a Gaussian Process Regressor with the optimized periodicity
gp = GaussianProcessRegressor(kernel=kernel(optimal_periodicity), n_restarts_optimizer=10, random_state=42)

# Fit the Gaussian Process to the data
gp.fit(X, y)

# Make predictions on new data
X_new = np.linspace(0, 5, 200)[:, np.newaxis]
y_pred, sigma = gp.predict(X_new, return_std=True)

# Plot the results
plt.figure(figsize=(8, 4))
plt.scatter(X, y, c='r', marker='.', label='Training Data')
plt.plot(X_new, y_pred, 'b-', label='GP Prediction')
plt.fill_between(X_new.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='blue', label='95% Confidence Interval')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Gaussian Process Regression with Periodic Kernel (Optimized Periodicity)')
plt.legend()
plt.savefig("MLE_vs_GP/GP.png")

print(f'Optimized Periodicity: {optimal_periodicity}')


######################### Markov-Chain Monte-Carlo #########################
import emcee
import corner

# Generate synthetic data
rng = np.random.default_rng(seed=42)
X = rng.uniform(0, 5, 20)
y = 3*np.sin(2*X + 0.5) + rng.normal(0, 0.1, len(X))

def log_likelihood(theta, x, y):
    A, omega, phi, sigma = theta
    model = A * np.sin(omega * x + phi)
    return -0.5 * np.sum((y - model) ** 2 / sigma**2 + np.log(sigma**2))

# Define the log prior
def log_prior(theta):
    A, omega, phi, sigma = theta
    if 0 < sigma < 10 and 0 < A < 5 and 0 < omega < 5 and -6 < phi < 6:
        return 0.0
    return -np.inf

# Define the log posterior
def log_posterior(theta, x, y):
    return log_prior(theta) + log_likelihood(theta, x, y)

def Model(x, theta):
    A, omega, phi, sigma = theta
    return A*np.sin(x*omega + phi)

def sample_walkers(domain,nsamples,flattened_chain):
    models = []
    draw = np.floor(np.random.uniform(0,len(flattened_chain),size=nsamples)).astype(int)
    thetas = flattened_chain[draw]
    for i in thetas:
        mod = Model(domain,i)
        models.append(mod)
    spread = np.std(models,axis=0)
    med_model = np.median(models,axis=0)
    return med_model, spread

# Function to plot the predicted lines
def plot_predicted_lines(sampler,x_range, med, spr, ax):
    samples = sampler.flatchain
    theta_max  = np.array(samples[np.argmax(sampler.flatlnprobability)])
    best_pred = Model(x_range,theta_max)
    ax.plot(x_range, best_pred, color = 'b',label='HL Parameters')
    ax.fill_between(x_range,med - spr, med + spr,color = 'b',alpha=0.2)
    ax.fill_between(x_range,med - spr*2, med + spr*2,color = 'b',alpha=0.2)
    ax.fill_between(x_range,med - spr*3, med + spr*3,color = 'b',alpha=0.2)


# Initial parameter values
initial_params = [0, 0, 0, 1]

# Set up the EnsembleSampler
nwalkers, ndim = 20, len(initial_params)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(X, y))

# Run initial burnin
print("Running Burn-in")
p0, _, _ = sampler.run_mcmc(np.random.rand(nwalkers, ndim) * 0.1 + initial_params, 200, progress=True)
sampler.reset()

# Run the MCMC sampling
print("Running production")
nsteps = 2000
pos, _, _ = sampler.run_mcmc(p0, nsteps, progress=True)

# Plot the posterior distributions
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain(discard=100,thin=100, flat=True)
labels = ['A', 'Omega', 'Phi','Sigma']
for i in range(ndim):
    ax = axes[i]
    ax.hist(samples[:, i], bins=50, color="k", alpha=0.5, histtype='step', density=True)
    ax.set_xlabel(labels[i])
    ax.set_yticks([])
fig.tight_layout()
plt.savefig("MLE_vs_GP/MCMC_Dist.png")

# Flatten the chain
samples_C = sampler.flatchain
fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.savefig("MLE_vs_GP/MCMC_Corner.png")

# Plot the predicted lines
x_range = np.linspace(0, 5, 100)
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the observed data
ax.scatter(X, y, color='red', label='Observations')

# Plot the predicted lines
Med, Spr = sample_walkers(x_range,200,samples_C)
plot_predicted_lines(sampler,x_range,Med, Spr, ax)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Posterior Predictive Plots')
ax.legend()
plt.savefig('MCMC_vs_GP/MCMC_Pred.png')

