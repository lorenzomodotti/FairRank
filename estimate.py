import numpy as np
import pickle
from utils import *
from config import *

'''
  Estimate mean and standard deviation of metrics
'''
def estimate(rho, ranking, B):
  # Relevance metric
  R = np.array([relevance_prob(rho[s], theta, mu) for s in range(S)], dtype = np.double)
  # Proportion of relevance
  RP = np.array([R[s] / np.sum(R[s]) for s in range(S)], dtype = np.double)
  # Mean long-run proportion (w.r.t T) of bookings
  b_emp = np.zeros((S,V), dtype = np.double)
  b_emp_std = np.zeros((S,V), dtype = np.double)
  b_emp_M = np.zeros((S,M,V))
  # Mean long-run proportion (w.r.t. total bookings) of bookings
  BP = np.zeros((S,V), dtype = np.double)
  BP_std = np.zeros((S,V), dtype = np.double)

  for s in range(S):
    B_vec = np.zeros((M,V))
    for m in range(M):
      for t in range(T):
        for k in range(K):
          v = ranking[s,m,t,k]
          #pos[s,m,t,v] = k
          BP[s,v] += B[s,m,t,k]
          B_vec[m,v] += B[s,m,t,k]
      b_emp_M[s,m] = B_vec[m] / T
      B_vec[m] /= np.sum(B_vec[m])
    b_emp[s] = (BP[s] / T)
    b_emp_std[s] = np.std(b_emp_M[s])
    BP[s] /= np.sum(BP[s])
    BP_std[s] = np.std(B_vec, axis = 0)

  psi_M = np.zeros((S,M,V), dtype = np.double)
  psi = np.zeros((S,V), dtype = np.double)
  psi_std = np.zeros((S,V), dtype = np.double)
  PSI = np.zeros(S, dtype = np.double)
  PSI_std = np.zeros(S, dtype = np.double)

  for s in range(S):
    for m in range(M):
      psi_M[s,m] = compute_psi(R[s], b_emp_M[s,m], V)
      psi[s] += psi_M[s,m]
    psi_std[s] = np.std(psi_M[s], axis = 0)
  psi /= M


  for s in range(S):
    PSI_vec = np.zeros(M)
    for m in range(M):
      PSI_vec[m] = compute_PSI(psi_M[s][m])
      PSI[s] += PSI_vec[m]
    PSI_std[s] = np.std(PSI_vec)
  PSI /= M

  output_dict = {'R' : R, 
                 'RP' : RP, 
                 'b_emp' : b_emp, 
                 'b_emp_std' : b_emp_std, 
                 'BP' : BP, 
                 'BP_std' : BP_std, 
                 'psi' : psi, 
                 'psi_std' : psi_std, 
                 'PSI' : PSI, 
                 'PSI_std' : PSI_std}
  if save:
    with open(files_path + 'estimates.pkl', 'wb') as f:
      pickle.dump(output_dict, f)
  return(output_dict)


'''
  Compute the fairness metric for a given item
'''
def compute_psi(R, b_bar, V):
  psi = np.zeros(V, dtype = np.double)
  for v in range(V):
    psi[v] = (np.sum(R)*b_bar[v]) - (R[v] * np.sum(b_bar))
  return(psi)

'''
  Compute the fairness metric for the platform
'''
def compute_PSI(psi):
  return(0.5 * np.sum(psi**2))