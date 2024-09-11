import numpy as np
from utils import rank_items
from config import *

'''
  Simulate the user choices under the Dynamic Bayesian Network Click model
'''
def user_behaviour_dbn(r_u, r_u_prime, gamma, K, rng):
  # Users examine items sequentially; item in position k is clicked w.p. r_u_prime[k] and then booked w.p. r_u[k]. If so, the user stops browsing.
  # If an item is not booked or clicked, the following is examined w.p. gamma. If an item is not examined the user stops browsing.
  # Vector of clicks
  clicks = np.zeros(K)
  # Vector of bookings
  books = np.zeros(K)
  for k in range(K):
    if rng.binomial(1, r_u_prime[k]): # User perceives item as relevant and clicks
      clicks[k] = 1
      if rng.binomial(1, r_u[k]): # Item is actually relevant and is booked
        books[k] = 1
        return(clicks, books, -2)
      else: # Item is not relevant and is not booked
        if rng.binomial(1, 1-gamma): # User stops browsing
          return(clicks, books, k)
    else: # User perceives item as non-relevant and does not click
      if rng.binomial(1, 1-gamma): # User stops browsing
          return(clicks, books, k)
  return(clicks, books, -1)

'''
  Update the user ranking re-adding an itme when it replenishes
'''
def update_ranking_dbn(rel, rel_prime, N, V, U):
  new_rel = np.copy(rel)
  new_rel_prime = np.copy(rel_prime)
  for v in range(V):
    if(N[v] == 0):
      new_rel[:,v] = np.zeros(U)
      new_rel_prime[:,v] = np.zeros(U)
  # Naive ranking: rk[u,k] is the item in position k of u's ranking
  new_rk = rank_items(new_rel, new_rel_prime)
  # Ordered relevance probability: r[u,k] is the relevance probability of the item in position k of u's ranking
  new_r =  np.take_along_axis(new_rel, new_rk, axis = 1)
  # Ordered perceived relevance probability: r_prime[u,k] is the perceived relevance probability of the item in position k of u's ranking
  new_r_prime = np.take_along_axis(new_rel_prime, new_rk, axis = 1)
  return(new_r, new_r_prime, new_rk)

'''
  Compute examination probabilities for a given user and ranking
'''
def compute_examination_probability(r_u, r_u_prime, gamma, K):
  epsilon = np.zeros(K)
  epsilon[0] = 1
  for k in range(K-1):
    if(r_u_prime[k+1] == 0):
      epsilon[k+1] = 0
    else:
      epsilon[k+1] = epsilon[k] * gamma * (r_u_prime[k] * (1-r_u[k]) + (1-r_u_prime[k]))
  return(epsilon)

'''
  Compute examination probabilities for each user and ranking under every crowding level
'''
def compute_epsilon(alpha, rel_prime_1, rel_prime_2, rel_1, rel_2, rk, gamma, K, U):
  epsilon = np.zeros((S,U,K))
  for s in range(S):
    rel_prime = (1-alpha[s]) * rel_prime_1 + alpha[s] * rel_prime_2
    rel = (1-alpha[s]) * rel_1 + alpha[s] * rel_2
    r = np.take_along_axis(rel, rk[s], axis = 1)
    r_prime = np.take_along_axis(rel_prime, rk[s], axis = 1)
    for u in range(U):
      epsilon[s,u,:] = compute_examination_probability(r[u], r_prime[u], gamma, K)
  return(epsilon)

'''
  Simulate the evolution of the platform
'''
def simulate_dbn(U, V, K, N0, mu_inv, r0, r_prime0, rel, rel_prime, rk0, gamma, theta, T, seed_number = 224):
  rng = np.random.default_rng(seed_number)
  # User type u sampled for each request
  user_ids = rng.choice(U, size = T, p = theta)
  # Initial r
  r = np.copy(r0)
  # Initial r_prime
  r_prime = np.copy(r_prime0)
  # Initial rk
  rk = np.copy(rk0)
  # Initial capacity
  N = np.copy(N0)
  # Remaining time before replenishment
  eta = np.zeros(V)
  # Ranking shown in each period
  ranking = np.zeros((T,K), dtype = np.int16)
  # Examination probability in each period
  epsilon = np.zeros((T,K))
  # Clicks for each position in each period
  C = np.zeros((T,K), dtype = np.int16)
  # Bookings for each position in each period
  B = np.zeros((T,K), dtype = np.int16)
  # Position at which the user stops browsing (last position examined)
  position_stop = np.zeros(T, dtype = np.int16)
  # Number of available items
  N_available = V
  # Number of 'up' periods (item v available)
  UP = np.zeros(V)
  # Utility collected
  utility = 0.0
  # Platform dynamics
  for t in range(T):
    # Indicator if an item replenished in this period to update rankings
    has_replenished = 0
    # Update replenishment times and capacities
    for v in range(V):
      # Item replenished in current period
      if eta[v] == 1:
        N[v] += 1
        eta[v] = 0
        has_replenished = 1
        N_available += 1
      # Item is still booked in current period
      elif eta[v] > 1:
        eta[v] -= 1
      if N[v] > 0:
        UP[v] += 1
    if has_replenished:
      r, r_prime, rk = update_ranking_dbn(rel, rel_prime, N, V, U)
    # If all items reached capacity skip iteration
    if N_available == 0:
        continue
    # Selected user to make requests
    u = user_ids[t]
    # Ranking shown
    ranking[t] = rk[u]
    # Examination probabilities faced
    epsilon[t] = compute_examination_probability(r[u], r_prime[u], gamma, K)
    # Simulate user behaviour according to the DBN
    c, b, p_s = user_behaviour_dbn(r[u], r_prime[u], gamma, K, rng)
    # Booked items
    booked_items = rk[u][b == 1]
    # Update clicks
    C[t][c == 1] += 1
    # Update bookings
    B[t][b == 1] += 1
    # Update utility
    utility += np.sum(r[u][b == 1])
    # Update position_stop
    position_stop[t] = p_s
    # Update capacity
    N[booked_items] -= 1
    # Update replenishment times
    eta[booked_items] += rng.geometric(1/mu_inv[booked_items]) + 1
    #eta[booked_items] += mu_inv[booked_items] + 1
    # Indicator if an item reached capacity in this period to update rankings
    has_reached_capacity = 0
    # Remove items that reached capacity
    for item in booked_items:
      if(N[item] == 0):
        has_reached_capacity = 1
        N_available -= 1
    if has_reached_capacity:
      r, r_prime, rk = update_ranking_dbn(rel, rel_prime, N, V, U)
  return(user_ids, ranking, epsilon, C, B, position_stop, UP, utility)