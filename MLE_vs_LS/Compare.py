################### HEADER ###################
# Author: Patrick Sandoval                   #
# Date: 2023-12-03                           #
# ############################################
# The following script 'fits' randomly       #
# generated data using least-squares         #
# and a Markov-Chain Monte-Carlo method      #
##############################################

##################### Least Squares #####################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# Generate synthetic data
rng = np.random.default_rng(seed=42)
X = np.sort(rng.uniform(0,5,30))
A, B, C, D = rng.random(4)
y = A*np.sin(X*B + C) + D + rng.normal(0, 0.1, len(X))
yerr = np.random.rand(len(X))*0.1
print(A,B,C,D)

# Look at generated data
fig, ax = plt.subplots(figsize=(8,4))
ax.errorbar(X,y,yerr=yerr,fmt='k.')
ax.set_xlabel('X',fontsize=16)
ax.set_ylabel('Y',fontsize=16)
ax.set_title('Synthetic Data',fontsize=18)
plt.tight_layout()
plt.savefig('MLE_vs_LS/Figures/SyntheticData.png')
plt.show()

# Define model function for curve_fit
def f(x, A, omega, phi, C):
    return A*np.sin(x*omega + phi) + C

# Define initial parameters 
guess_rng = np.random.default_rng(seed=1)
p0 = guess_rng.random(4)

popt, pcov = curve_fit(f, X, y, sigma=yerr, p0=p0, maxfev=2000)

X_fine = np.linspace(X.min(), X.max(), 200)

fig, ax = plt.subplots(figsize=(10,8))
labels = ['A',r'$\omega$',r'$\Phi$','C']
sns.heatmap(pcov,annot=True,cmap='BuPu',xticklabels=labels,yticklabels=labels,ax=ax)
ax.set_title("Covariance Matrix for L.S. Parameters",fontsize=18)
ax.set_yticklabels(labels,rotation=0,fontsize=16)
ax.set_xticklabels(labels,fontsize=16)
plt.savefig("MLE_vs_LS/Figures/NACovMat.png")
plt.show()

print("Best Fit Parameters from LS:")
print("A:",round(popt[0],3),"(+/-)",round(np.sqrt(np.diag(pcov)[0]),3))
print("Omega:",round(popt[1],3),"(+/-)",round(np.sqrt(np.diag(pcov)[1]),3))
print("Phi:",round(popt[2],3),"(+/-)",round(np.sqrt(np.diag(pcov)[2]),3))
print("C:",round(popt[3],3),"(+/-)",round(np.sqrt(np.diag(pcov)[3]),3))
print()
################## dunia is cool ##################

################## Maximize Likelihood ##################
from scipy.optimize import minimize
def Model(x, theta):
    A, omega, phi, C, log_f = theta
    return A*np.sin(x*omega + phi) + C

# Def log likelihhod
def log_likelihood(theta, x, y, yerr):
    ''' This definition of the log likelihood function
    incorportate the fractional error from the model'''
    log_f = theta[-1]
    model = Model(x,theta)
    sigma2 = yerr**2 + np.exp(2 * log_f) * model**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

##########################################################################
# def log_likelihood(theta, x, y, yerr):                                 #
#     ''' This definition of the likelihood function                     #
#     yields the same as LS'''                                           #
#     A, omega, phi, C, sigma = theta                                    #
#     model = A * np.sin(omega * x + phi) + C                            #
#     return -0.5 * np.sum((y - model) ** 2 / yerr**2 + np.log(yerr**2)) #
##########################################################################

# Set initial params same as LS and add log f
initial_params = np.append(p0, -1)

# Define parameter bounds
bounds = [(0, 1.1), (0, 1.1), (0, 1.1), (0, 1.1), (None, None)]

# Define negative ll
nll = lambda *args: -log_likelihood(*args)

# Minimize the negative log likelihood
soln = minimize(nll, x0=initial_params, args=(X,y,yerr), bounds=bounds, method='Nelder-Mead',tol=1e-16,options={'maxfev':2000})

# Collect optimal parameters
A_ml, omega_ml1, phi_ml1 , C_ml, log_f_ml = soln.x
print("Maximzing log likelihood:")
print(soln.success)
print(soln.message,'\n')

print("Maximum Likelihood Estimates:")
print("A = {0:.3f}".format(A_ml))
print("Omega = {0:.3f}".format(omega_ml1))
print("Phi = {0:.3f}".format(phi_ml1))
print("C = {0:.3f}".format(C_ml))
print("Log f = {0:.3f}".format(log_f_ml))


# Plot prediction
fig, (a0, a1) = plt.subplots(figsize=(8,6),nrows=2,gridspec_kw={'height_ratios': [3, 1]},sharex=True)
a0.plot(X_fine, f(X_fine,*popt),c = 'b',ls='--',label='Least-Squares',alpha=0.8)
a0.set_ylabel("Y",fontsize=16)
#a0.plot(X_fine,Model(X_fine,soln.x),c='hotpink',ls='-.',label='Maximum-Likelihood')
a0.errorbar(X, y ,yerr = yerr, fmt = 'k.',label='Data')
a0.plot(X_fine,A*np.sin(X_fine*B + C) + D,c='k',label="True")
a0.set_title("Least-Square Predictive Plot",fontsize=18)
a0.legend()

a1.set_xlabel("X",fontsize=16)
a1.set_ylabel(r"Y - $f$(X,$\theta$)",fontsize=16)
a1.scatter(X, y - f(X,*popt), c='b',marker='.')
#a1.scatter(X, y - Model(X,soln.x), c='hotpink',marker='.')
plt.savefig("MLE_vs_LS/Figures/Prediction Comparison.png")
plt.show()

######################### Sample the posterior with MCMC #########################
import emcee
import corner 

# Def prior to avoid bimodalities in posterior dist
def log_prior(theta):
    A, omega, phi, C, sigma = theta
    if 0 <= phi < 1.1 and 0 <= omega < 1.1 and 0 <= A < 1.1 and 0 <= C < 1.1:
         return 0.0
    return -np.inf

# Def the log posterior to sample from
def log_posterior(theta, x, y, yerr):
    return log_prior(theta) + log_likelihood(theta, x, y, yerr)

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
    ax.plot(x_range, best_pred, color = 'k', ls = ':',label='HL Parameters')
    ax.fill_between(x_range,med - spr, med + spr,color = 'r',alpha=0.2)
    ax.fill_between(x_range,med - spr*2, med + spr*2,color = 'r',alpha=0.2)
    ax.fill_between(x_range,med - spr*3, med + spr*3,color = 'r',alpha=0.2)

# Set up the EnsebleSampler
nwalkers, ndim = 60, len(initial_params)
sampler = emcee.EnsembleSampler(nwalkers,ndim, log_posterior, args=(X,y,yerr))

# Run initial burn-in
print("Running Burn-in")
p0, _, _, = sampler.run_mcmc(np.random.rand(nwalkers,ndim) * 0.1 + initial_params, 100, progress=True)
sampler.reset()

# Run the MCMC sampling
print("Running production")
nsteps = 2000
pos, _, _ = sampler.run_mcmc(p0, nsteps, progress=True)

# Flatten the chain
samples = sampler.flatchain
# Sample the flatten chain
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
# Add log f to labels
labels.append("log f")

# Plot corner plot
fig = corner.corner(flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
#plt.savefig("MLE_vs_LS/Figures/MCMC_Corner.png")
plt.show()

# Plot the predicted lines
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(X,y,yerr=yerr,fmt='k.')
ax.plot(X_fine, A*np.sin(X_fine*B + C) + D, color='gray',linewidth=3,label='True',alpha=0.5)
Med, Spr = sample_walkers(X_fine, 500, flat_samples)
plot_predicted_lines(sampler, X_fine , Med, Spr, ax)
ax.plot(X_fine, f(X_fine,*popt),ls='--',c='g',label='Least-Square')
ax.set_xlabel('X',fontsize=16)
ax.set_ylabel('Y',fontsize=16)
ax.set_title('Posterior Predictive Plots',fontsize=18)
ax.legend()
#plt.savefig('MLE_vs_LS/Figures/MCMC_Pred.png')
plt.show()


# Print the best parameters
theta_max  = np.array(samples[np.argmax(sampler.flatlnprobability)])
print()
print("MCMC MLE Results")
print("A:",theta_max[0])
print("Omega:",theta_max[1])
print("Phi:",theta_max[2])
print("C:",theta_max[3])
print("Log(f):",theta_max[4])

