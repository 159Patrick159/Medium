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
def SquaredExponential(x1,x2,a,l):
    -1/2/l * scipy.spatial.distance.cdist(x1,x2,'sqeuclidean')