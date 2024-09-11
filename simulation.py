from datetime import datetime
import time
import pytz
import pickle
import numpy as np
from config import *
from utils import *
from dbn import *

'''
    Simulate the platform for different levels of crowding
'''
def simulate():

  # Simulation ID
  simulation_time = datetime.now(pytz.timezone('America/New_York')).strftime("%Y-%m-%d_%H-%M")
  simulation_id = 'simulation_' + simulation_time

  # Save simulation parameters
  if save:
    with open(files_path + 'parameters_' + simulation_id + '.pkl', 'wb') as f:
        pickle.dump([V, U, K, N0, mu, rel, rel_prime, rk, r, r_prime, gamma, theta, T, S, M, q, z], f)

  # Outputs
  rho = np.zeros((S,U,V), dtype = np.double)
  rk = np.zeros((S,U,K), dtype = np.int16)
  pos = np.zeros((S,U,V), dtype = np.int16)
  user_id = np.zeros((S,M,T), dtype = np.int16)
  ranking = np.zeros((S,M,T,K), dtype = np.int16)
  epsilon = np.zeros((S,M,T,K))
  C = np.zeros((S,M,T,K), dtype = np.int16)
  B = np.zeros((S,M,T,K), dtype = np.int16)
  position_stop = np.zeros((S,M,T), dtype = np.int16)
  UP = np.zeros((S,M,V))
  utility = np.zeros((S,M))

  print('- Simulation {} -\n'.format(simulation_time))
  start_time = time.time()

  # Repeat for each level of crowding
  for s in range(S):
    print(" > Simulation, alpha = {}".format(alpha[s]))
    # Perceived relevance probability: rel_prime[u,v] is the perceived relevance probability of item v for user u
    rel_prime_1 = generate_uniform_relevance(U, V, rng = rng)
    rel_prime_2 = generate_random_crowded_linear_relevance(U, V, rng = rng)
    rel_prime = (1-alpha[s]) * rel_prime_1 + alpha[s] * rel_prime_2
    rel = (1-alpha[s]) * const_rel * rel_prime_1 + alpha[s] * const_rel * rel_prime_2
    # Effective relevance probability: rho[u,v] = rel_prime[u,v] * rel[u,v]
    rho[s] = np.multiply(rel, rel_prime)
    # Naive ranking: rk[u,k] is the item in position k of u's ranking
    rk[s] = rank_items(rel, rel_prime)
    # Naive ranking: pos[u,v] is the position of item v in u's ranking
    pos[s] = get_ranking_items(rk[s])
    # Ordered relevance probability: r[u,k] is the relevance probability of the item in position k of u's ranking
    r = np.take_along_axis(rel, rk[s], axis = 1)
    # Ordered perceived relevance probability: r_prime[u,k] is the perceived relevance probability of the item in position k of u's ranking
    r_prime = np.take_along_axis(rel_prime, rk[s], axis = 1)
    # Repeat simulations
    for m in range(M):
      user_id[s][m], ranking[s][m], epsilon[s][m], C[s][m], B[s][m], position_stop[s][m], UP[s][m], utility[s][m] = simulate_dbn(U, V, K, N0, mu_inv, r, r_prime, rel, rel_prime, rk[s], gamma, theta, T, m)
  end_time = time.time()
  print("- Total run time:  {} seconds -\n".format(end_time - start_time))

  output_dict = {'rho' : rho,
                 'rk' : rk,
                 'pos' : pos, 
                 'user_id' : user_id,
                 'ranking' : ranking, 
                 'epsilon' : epsilon, 
                 'C' : C, 
                 'B' : B,
                 'position_stop' : position_stop, 
                 'UP' : UP, 
                 'utility' : utility}
  # Save simulation output
  if save:
    with open(files_path + 'output_' + simulation_id + '.pkl', 'wb') as f:
        pickle.dump(output_dict, f)
  return(output_dict)




'''
    Return the parameters of the platform for different levels of crowding
'''
def get_sim_parameters():

  # Simulation setup
  rel_prime_1 = np.zeros((U,V), dtype = np.double)
  rel_prime_2 = np.zeros((U,V), dtype = np.double)
  rel_1 = np.zeros((U,V), dtype = np.double)
  rel_2 = np.zeros((U,V), dtype = np.double)
  rho = np.zeros((S,U,V), dtype = np.double)
  rk = np.zeros((S,U,K), dtype = np.int16)
  pos = np.zeros((S,U,V), dtype = np.int16)

  rel_prime_1 = generate_uniform_relevance(U, V, rng = rng)
  rel_prime_2 = generate_random_crowded_linear_relevance(U, V, rng = rng)
  rel_1 = const_rel*np.copy(rel_prime_1)
  rel_2 = const_rel*np.copy(rel_prime_2)

  for s in range(S):
    # Perceived relevance probability: rel_prime[u,v] is the perceived relevance probability of item v for user u
    rel_prime = (1-alpha[s]) * rel_prime_1 + alpha[s] * rel_prime_2
    rel = (1-alpha[s]) * rel_1 + alpha[s] * rel_2
    # Effective relevance probability: rho[u,v] = rel_prime[u,v] * rel[u,v]
    rho[s] = np.multiply(rel, rel_prime)
    # Naive ranking: rk[u,k] is the item in position k of u's ranking
    rk[s] = rank_items(rel, rel_prime)
    # Naive ranking: pos[u,v] is the position of item v in u's ranking
    pos[s] = get_ranking_items(rk[s])

    output_dict = {'rho' : rho, 
                   'rk' : rk, 
                   'pos' : pos,
                   'rel_prime_1' : rel_prime_1,
                   'rel_prime_2' : rel_prime_2,
                   'rel_1': rel_1,
                   'rel_2' : rel_2}
  return(output_dict)