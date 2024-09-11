import os
import numpy as np
from scipy.stats import norm

'''
    Parameter configuration
'''

# Paths
path = './test/'
plots_path = path + 'plots/'
files_path = path + 'files/'
if not os.path.exists(plots_path):
    os.makedirs(plots_path)
if not os.path.exists(files_path):
    os.makedirs(files_path)

# Whether to save results on file or not
save = False

# Random number generator
rng = np.random.default_rng(224)

# Items
V = 20

# Users
U = V

# Positions
K = V

# Capacity of items
const_cap = 1

# Initial capacity of items
N0 = const_cap*np.ones(V)

# Mean replenishment time of items
mu_inv = np.ones(V)
#mu_inv = rng.integers(low = 1, high = 3, size = V, endpoint = True)

# Replenishment rate
mu = 1/mu_inv

# Scale of booking probability w.r.t. click probability
const_rel = 0.4

# Continuation probability
gamma = 0.9

# Request probability for different user types
theta = np.array([1/U for _ in range(U)], dtype = np.double)

# Time horizon
T = 1000

# Number of runs per simulation
M = 5

# Quantile
quantile = 0.025
z = norm.ppf(1-quantile)

# Levels of crowding
alpha = np.linspace(0, 1, 6)

# Number of levels of crowding
S = len(alpha)

# Labels for plots
labels = [r'$\alpha = {}$'.format(np.round(alpha[s], 2)) for s in range(S)]