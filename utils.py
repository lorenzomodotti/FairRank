import numpy as np
from itertools import chain, repeat


'''
  Generate linear relevance structure
'''
def generate_linear_relevance(U, V, max_rel = 0.95, rng = None):
  rel = np.zeros((U,V))
  for u in range(U):
    for v in range(V):
      rel[u,v] = max_rel * (1 - abs(u-v)/V)
  return(rel)

'''
  Generate circular relevance structure
'''
def generate_circular_relevance(U, V, max_rel = 0.95, rng = None):
  rel = np.zeros((U,V))
  for u in range(U):
    for v in range(V):
      rel[u,v] = max_rel * (V - (u-v) % V) / V
  return(rel)

'''
  Generate radial relevance structure
'''
def generate_radial_relevance(U, V, max_rel = 0.95, rng = None):
  rel = np.zeros((U,V))
  for u in range(U):
    for v in range(V):
      rel[u,v] = max_rel * ( ((1 + u % U) + (1 + v % V)) / (U+V) )
  return(rel)

'''
  Generate crowded linear relevance structure
'''
def generate_crowded_linear_relevance(U, V, max_rel = 0.95, rng = None):
  rel = np.zeros((U,V))
  for u in range(U):
    for v in range(V):
      rel[u,v] = max_rel * ( (v+1) / V )
  return(rel)

'''
  Generate random crowded linear relevance structure
'''
def generate_random_crowded_linear_relevance(U, V, max_rel = 0.95, rng = None):
  bins_lower = np.zeros(10)
  bins_lower[0] = 1 - max_rel
  bins_upper = np.zeros(10)
  bins_upper[-1] = max_rel
  bins_lower[1:] = np.array([max_rel * (j+1) / 10 for j in range(9)])
  bins_upper[:-1] = np.array([max_rel * (j+2) / 10 for j in range(9)])
  rel = np.zeros((U,V))
  indices = list(chain.from_iterable([list(repeat(j, V//10)) for j in range(10)]))
  for u in range(U):
    for v in range(V):
      rel[u,v] = rng.uniform(bins_lower[indices[v]], bins_upper[indices[v]])
  return(rel)

'''
  Generate uniform relevance structure
'''
def generate_uniform_relevance(U, V, max_rel = 0.95, rng = None):
  rel = np.zeros((U,V))
  for u in range(U):
    rel[u] = rng.uniform(1-max_rel, max_rel, size = V)
  return(rel)

'''
  Compute the probability of being relevant
'''
def relevance_prob(rho, theta, mu):
  return(theta.dot(rho*mu))

'''
  Relevance-based ranking of items
'''
def rank_items(rel, rel_prime):
    return(np.argsort(-np.multiply(rel, rel_prime), axis = 1))

'''
  Return position of items in the ranking
'''
def get_ranking_items(rk):
  return(np.argsort(rk, axis = 1))