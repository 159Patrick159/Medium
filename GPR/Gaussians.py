################### HEADER ###################
# Author: Patrick Sandoval                   #
# Date: 2024-04-15                           #
# ############################################
# The following script generate 1D and 2D    #
# Gaussians and plots them for illustrative  #
# purposes for the Medium post.              #
##############################################

########## Relevant Imports ##########
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Define variables for Gaussians
N = 80
x = np.linspace(-5,5,N)
y = np.linspace(-5,5,N)
# Define meshgrid variables for 2D Gaussian
X, Y = np.meshgrid(x, y)

#Define params for Gaussian
mu = np.array([0,0])
sigma = np.array([[1,-0.6],[-0.6,1]])

# Define function to calculate MVN distribution
def multivariate_gaussian(pos, mu, Sigma):
    '''
    ###################### Inputs ######################
    # pos: Grid array of coordinates for plotting      #
    # mu: n-dim mean vector defining means             #
    # Sigma: nxn-dim covariance matrix                 #
    ###################### Output ######################
    # Multivariate gaussian matching the grid          #
    # dimensions of pos varible with mean mu and       #
    # kernel Sigma                                     #
    ####################################################
    '''
    # Define number of dimensions
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    # Compute normalization constant for MVN
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    # Use einstein summation convention to compute terms inside exponent
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)
    return np.exp(-fac / 2) / N

# Create grid array for plotting
pos = np.empty(X.shape + (2,))
# Set the first dim to X grid
pos[:,:,0] = X
# Set second dim to Y grid
pos[:,:,1] = Y
# So pos[i,j] = [X[i,j],Y[i,j]]

# Compute MVN for X and Y given mu and Sigma
Z = multivariate_gaussian(pos,mu=mu,Sigma=sigma)
# Compute marginal distributions for X and Y
zxy = Z[Z.shape[0]//2,:]
zyx = Z[:,Z.shape[0]//2]

fig = plt.figure()
ax1 = fig.add_subplot(projection='3d')
ax1.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
cset = ax1.contourf(X, Y, Z, zdir='z', offset=-0.2, cmap=cm.viridis)
cbar = plt.colorbar(
    cset, ax=ax1, fraction=0.025, pad=0.05)
pxy = ax1.plot(x,4*np.ones(len(x)),zxy,label=r"P(X$|$Y=0)")
pyx = ax1.plot(-4*np.ones(len(x)),y,zxy,label=r"P(Y$|$X=0)")
ax1.set_xlabel("X",fontsize=14)
ax1.set_ylabel("Y",fontsize=14)
cbar.ax.set_ylabel('$P(X,Y)$', fontsize=10)
ax1.set_zlim(-0.2,0.2)
ax1.legend()
ax1.view_init(elev=20,azim=-56,roll=0)
plt.tight_layout()
plt.savefig("GPR/Figures/BVN.png")