import numpy as np
from utils import *
  
'''
  Solve the system of implicit equations to find the equilibrium
'''
def implicit_equations(q0, mu, theta, rho, gamma, rk, pos, U, V):
  q = q0
  f = np.zeros(V, dtype = np.double)
  for v in range(V):
    f[v] += mu[v]
    sum_v = mu[v]
    for u in range(U):
      prod_u = theta[u] * rho[u,v]
      for k in range(pos[u,v]):
        prod_u *= q[rk[u,k]] * gamma * (1 - rho[u,rk[u,k]]) + (1 - q[rk[u,k]])
      sum_v += prod_u
    f[v] -= q[v] * sum_v
  return(f)

'''
  Compute the Jacobian of F w.r.t. q
'''
def jacobian_F_q(q, mu, theta, rho, gamma, rk, pos, U, V):
  J = np.zeros((V,V), dtype = np.double)
  for v in range(V):
    J[v,v] = -mu[v]
    for u in range(U):
      prod_u = theta[u] * rho[u,v]
      for k in range(rk[u,v]):
        prod_u *= q[rk[u,k]] * gamma * (1 - rho[u,rk[u,k]]) + (1 - q[rk[u,k]])
      J[v,v] -= prod_u
    #J[v,v] = -mu[v] / q[v]
    for w in range(v):
      sum_w = 0
      for u in range(U):
        if(rk[u,w] < rk[u,v]):
          prod_u = theta[u] * rho[u,v] * (gamma * (1 - rho[u,w]) - 1)
          for k in range(pos[u,w]):
            prod_u *= q[rk[u,k]] * gamma * (1 - rho[u,rk[u,k]]) + (1 - q[rk[u,k]])
          for k in range(pos[u,w]+1, pos[u,v]):
            prod_u *= q[rk[u,k]] * gamma * (1 - rho[u,rk[u,k]]) + (1 - q[rk[u,k]])
          sum_w += prod_u
      J[v,w] = - q[v] * sum_w
    for w in range(v+1,V):
      sum_w = 0
      for u in range(U):
        if(rk[u,w] < rk[u,v]):
          prod_u = theta[u] * rho[u,v] * (gamma * (1 - rho[u,w]) - 1)
          for k in range(pos[u,w]):
            prod_u *= q[rk[u,k]] * gamma * (1 - rho[u,rk[u,k]]) + (1 - q[rk[u,k]])
          for k in range(pos[u,w]+1, pos[u,v]):
            prod_u *= q[rk[u,k]] * gamma * (1 - rho[u,rk[u,k]]) + (1 - q[rk[u,k]])
          sum_w += prod_u
      J[v,w] = - q[v] * sum_w
  return(J)

'''
  Compute the Jacobian of rho w.r.t. alpha
'''
def jacobian_rho_alpha(alpha,  rel_prime_1, rel_prime_2, rel_1, rel_2, U, V):
  J = np.zeros((U,V), dtype = np.double)
  for u in range(U):
    for v in range(V):
      J[u,v] = rel_prime_1[u,v] * rel_2[u,v] + rel_prime_2[u,v] * rel_1[u,v] - 2 * rel_prime_1[u,v] * rel_1[u,v] \
              + 2 * alpha * (rel_prime_1[u,v] * rel_1[u,v] - rel_prime_1[u,v] * rel_2[u,v] - rel_prime_2[u,v] * rel_1[u,v] + rel_prime_2[u,v] * rel_2[u,v])
  return(J)

'''
  Compute the Jacobian of F w.r.t. alpha
'''
def jacobian_F_alpha(q, theta, rho, gamma, J_rho_alpha, rk, pos, U, V):
  J = np.zeros(V, dtype = np.double)
  for v in range(V):
    sum_1 = 0.0
    sum_2 = 0.0
    for u in range(U):
      prod_u = theta[u] * J_rho_alpha[u,v]
      for k in range(pos[u,v]):
        prod_u *= q[rk[u,k]] * gamma * (1 - rho[u,rk[u,k]]) + (1 - q[rk[u,k]])
      sum_1 += prod_u
    for u in range(U):
      J_prod = 0.0
      for j in range(pos[u,v]):
        prod_j = q[rk[u,j]] * gamma * J_rho_alpha[u,rk[u,j]]
        for k in range(j):
          prod_j *= q[rk[u,k]] * gamma * (1 - rho[u,rk[u,k]]) + (1 - q[rk[u,k]])
        for k in range(j+1, pos[u,v]):
          prod_j *= q[rk[u,k]] * gamma * (1 - rho[u,rk[u,k]]) + (1 - q[rk[u,k]])
        J_prod -= prod_j
      sum_2 += theta[u] * rho[u,v] * J_prod
    J[v] = - q[v] * (sum_1 + sum_2)
  return(J)

'''
  Compute the Jacobian of psi w.r.t. alpha
'''
def jacobian_psi_v_alpha(q, mu, theta, rho, J_rho_alpha, J_q_alpha, U, V):
  J = np.zeros(V, dtype = np.double)
  for v in range(V):
    for u in range(U):
      sum_u = 0.0
      for w in range(v):
        sum_u += mu[v] * (J_rho_alpha[u,w] * (1-q[v]) - rho[u,w] * J_q_alpha[v]) - mu[w] * (J_rho_alpha[u,v] * (1-q[w]) - rho[u,v] * J_q_alpha[w])
      for w in range(v+1, V):
        sum_u += mu[v] * (J_rho_alpha[u,w] * (1-q[v]) - rho[u,w] * J_q_alpha[v]) - mu[w] * (J_rho_alpha[u,v] * (1-q[w]) - rho[u,v] * J_q_alpha[w])
      J[v] += theta[u] * sum_u
  return(J)

'''
  Compute the Jacobian of PSI w.r.t. alpha
'''
def jacobian_PSI_alpha(psi, J_psi_alpha, V):
  J = 0.0
  for v in range(V):
      J += psi[v] * J_psi_alpha[v]
  return(J)