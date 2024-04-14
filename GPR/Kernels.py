import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import scipy.spatial
import seaborn as sns

# Set seaborn dark style grid
sns.set_style('darkgrid')

# Define kernels through functions
def SquaredExponential(x1,x2,a=1,l=1):
    '''
    Definition of Squared Exponantial or RBF kernel with 
    hyperparamters a (vertical scale) and l (horizontal scale)
    '''
    val = -1/2/l**2 * scipy.spatial.distance.cdist(x1,x2,'sqeuclidean')
    return a**2*np.exp(val)

def RationalQuadratic(x1,x2,alpha=1,l=1):
    return (1 + scipy.spatial.distance.cdist(x1,x2,'sqeuclidean')/(2*alpha*l**2))**(-alpha)

def ExpSine(x1,x2,a=1,l=1,p=1):
    '''
    Definition of Sine Squared Exponential kernel with 
    hyperparameters a:(vertical scale), l:(length scale)
    and p:(periodicity)
    '''
    val = -2*np.sin(np.pi*scipy.spatial.distance.cdist(x1,x2,'sqeuclidean')/p)**2/l**2
    return a**2*np.exp(val)

# Plot all kernels
fig, ax = plt.subplots(3,2, figsize=(8, 11))
xlim = (-3, 3)
X = np.expand_dims(np.linspace(*xlim, 50), 1)
sigma = SquaredExponential(X, X,)
# Plot covariance matrix
im = ax[0,0].imshow(sigma)
cbar = plt.colorbar(
    im, ax=ax[0,0], fraction=0.045, pad=0.05)
cbar.ax.set_ylabel('$k(x,x)$', fontsize=10)
ax[0,0].set_title((
    'Square Exponential \n'
    'example of covariance matrix'))
ax[0,0].set_xlabel('x', fontsize=13)
ax[0,0].set_ylabel('x', fontsize=13)
ticks = list(range(xlim[0], xlim[1]+1))
ax[0,0].set_xticks(np.linspace(0, len(X)-1, len(ticks)))
ax[0,0].set_yticks(np.linspace(0, len(X)-1, len(ticks)))
ax[0,0].set_xticklabels(ticks)
ax[0,0].set_yticklabels(ticks)
ax[0,0].grid(False)

# Show covariance with X=0
zero = np.array([[0]])
sigma0 = SquaredExponential(X, zero)
# Make the plots
ax[0,1].plot(X[:,0], sigma0[:,0], label='$k(x,0)$')
ax[0,1].set_xlabel('x', fontsize=13)
ax[0,1].set_ylabel('covariance', fontsize=13)
ax[0,1].set_title((
    'Square Exponential covariance\n'
    'between $x$ and $0$'))
ax[0,1].set_xlim(*xlim)
ax[0,1].legend(loc=1)

sigmaSine = ExpSine(X,X,p=3)

# Plot covariance matrix
im = ax[1,0].imshow(sigmaSine)
cbar = plt.colorbar(
    im, ax=ax[1,0], fraction=0.045, pad=0.05)
cbar.ax.set_ylabel('$k(x,x)$', fontsize=10)
ax[1,0].set_title((
    'Exponential Squared Sine \n'
    'example of covariance matrix'))
ax[1,0].set_xlabel('x', fontsize=13)
ax[1,0].set_ylabel('x', fontsize=13)
ticks = list(range(xlim[0], xlim[1]+1))
ax[1,0].set_xticks(np.linspace(0, len(X)-1, len(ticks)))
ax[1,0].set_yticks(np.linspace(0, len(X)-1, len(ticks)))
ax[1,0].set_xticklabels(ticks)
ax[1,0].set_yticklabels(ticks)
ax[1,0].grid(False)

# Show covariance with X=0
zero = np.array([[0]])
sigma0sine = ExpSine(X, zero)
# Make the plots
ax[1,1].plot(X[:,0], sigma0sine[:,0], label='$k(x,0)$')
ax[1,1].set_xlabel('x', fontsize=13)
ax[1,1].set_ylabel('covariance', fontsize=13)
ax[1,1].set_title((
    'Exponential Squared Sine covariance\n'
    'between $x$ and $0$'))
# ax2.set_ylim([0, 1.1])
ax[1,1].set_xlim(*xlim)
ax[1,1].legend(loc=1)

sigmaQuad = RationalQuadratic(X, X)
# Plot covariance matrix
im = ax[2,0].imshow(sigmaQuad)
cbar = plt.colorbar(
    im, ax=ax[2,0], fraction=0.045, pad=0.05)
cbar.ax.set_ylabel('$k(x,x)$', fontsize=10)
ax[2,0].set_title((
    'Rational Quadratic Exponential \n'
    'example of covariance matrix'))
ax[2,0].set_xlabel('x', fontsize=13)
ax[2,0].set_ylabel('x', fontsize=13)
ticks = list(range(xlim[0], xlim[1]+1))
ax[2,0].set_xticks(np.linspace(0, len(X)-1, len(ticks)))
ax[2,0].set_yticks(np.linspace(0, len(X)-1, len(ticks)))
ax[2,0].set_xticklabels(ticks)
ax[2,0].set_yticklabels(ticks)
ax[2,0].grid(False)

# Show covariance with X=0
zero = np.array([[0]])
sigma0Quad = RationalQuadratic(X, zero)
# Make the plots
ax[2,1].plot(X[:,0], sigma0Quad[:,0], label='$k(x,0)$')
ax[2,1].set_xlabel('x', fontsize=13)
ax[2,1].set_ylabel('covariance', fontsize=13)
ax[2,1].set_title((
    'Rational Quadratic covariance\n'
    'between $x$ and $0$'))
# ax2.set_ylim([0, 1.1])
ax[2,1].set_xlim(*xlim)
ax[2,1].legend(loc=1)

fig.tight_layout()
#plt.savefig("GPR/Figures/Kernel Comparison.png")
plt.show()