from simulation import get_sim_parameters
from estimate import *
from config import *
from dbn import *
from numerical import *
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

'''
  Get parameters
'''
print('- Get parameters -')
sim_parameters = get_sim_parameters()
rho = sim_parameters['rho'] 
rk = sim_parameters['rk'] 
pos = sim_parameters['pos']
rel_prime_1 = sim_parameters['rel_prime_1']
rel_prime_2 = sim_parameters['rel_prime_2']
rel_1 = sim_parameters['rel_1']
rel_2 = sim_parameters['rel_2']

R = np.array([relevance_prob(rho[s], theta, mu) for s in range(S)], dtype = np.double)
RP = np.array([R[s] / np.sum(R[s]) for s in range(S)], dtype = np.double)

'''
  Solve for equilibrium
'''
print('- Solve for equilibrium -')
q_th = np.array([fsolve(implicit_equations, 0.5*np.ones(V), args = (mu, theta, rho[s], gamma, rk[s], pos[s], U, V)) for s in range(S)], dtype = np.double)
b_th = np.multiply(mu, (1-q_th))
psi_th = np.array([compute_psi(R[s], b_th[s], V) for s in range(S)], dtype = np.double)
PSI_th = np.array([compute_PSI(psi_th[s]) for s in range(S)], dtype = np.double)

'''
  Solve for one-shot
'''
print('- Solve for one-shot -')
epsilon = compute_epsilon(alpha, rel_prime_1, rel_prime_2, rel_1, rel_2, rk, gamma, K, U)
b_0 = np.zeros((S,V))
for s in range(S):
  for v in range(V):
    b_0[s,v] = np.sum([theta[u] * rho[s,u,v] * epsilon[s,u,pos[s,u,v]] for u in range(U)])

psi_0 = np.array([compute_psi(R[s], b_0[s], V) for s in range(S)], dtype = np.double)
PSI_0 = np.array([compute_PSI(psi_0[s]) for s in range(S)], dtype = np.double)

'''
  Plot relevance barplot
'''
print('- Plot relevance -')
fig, axs = plt.subplots(2, 3, figsize=(18,12), sharey = True)
fig.add_subplot(111, frameon = False)
for s in range(3):
  items = np.argsort(RP[s])
  axs[0,s].plot(RP[s,items], color = 'orange')
  axs[0,s].plot(b_th[s,items], color = 'cornflowerblue', linestyle='dashed', linewidth = 1)
  axs[0,s].plot(b_0[s,items], color = 'black', linestyle='dashed', linewidth = 1)
  axs[0,s].set_title(labels[s], fontsize = 14, pad = 2)
  axs[0,s].set_xticks([])
for s in range(3,6):
  items = np.argsort(RP[s])
  axs[1,s-3].plot(RP[s,items], color = 'orange')
  axs[1,s-3].plot(b_th[s,items], color = 'cornflowerblue', linestyle='dashed', linewidth = 1)
  axs[1,s-3].plot(b_0[s,items], color = 'black', linestyle='dashed', linewidth = 1)
  axs[1,s-3].set_title(labels[s], fontsize = 14, pad = 2)
  axs[1,s-3].set_xticks([])
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.subplots_adjust(wspace=0, hspace=0)
rel_line = Line2D([0], [0], linestyle='solid', color='orange', label=r'$\mathcal{R}$')
eq_line = Line2D([0], [0], linestyle='dashed', color='cornflowerblue', label=r'$\bar{b}$ (equilibrium)')
os_line = Line2D([0], [0], linestyle='dashed', color='black', label=r'$\bar{b}$ (one-shot)')
plt.legend(handles=[eq_line, os_line, rel_line], bbox_to_anchor =(0.5,-0.1), loc='lower center', fontsize=16, ncol=3)
plt.tight_layout()
plt.savefig(plots_path + 'th_relevance.png', dpi = 1000)
plt.show()


b_bar_th = np.array([np.sum(b_th[s]) for s in range(S)])
b_bar_0 = np.array([np.sum(b_0[s]) for s in range(S)])

'''
  Plot bookings
'''
print('- Plot bookings -')
fig = plt.figure(1)
plt.plot(alpha, b_bar_th, color = 'cornflowerblue', linestyle=(0, (5, 5)), label = r'$\bar{b}$ (equilibrium)')
plt.plot(alpha, b_bar_0, color = 'black', linestyle=(0, (5, 5)), label = r'$\bar{b}$ (one-shot)')
plt.xlabel(r'$\alpha$', fontsize=16)
plt.xticks(alpha)
plt.legend(bbox_to_anchor =(0.5,-0.3), loc='lower center', ncol = 2, fontsize=12)
fig.savefig(plots_path + 'bookings.png', dpi = 1000, bbox_inches='tight')
plt.show()

'''
  Plot fairness metric
'''
print('- Plot fairness -')
fig = plt.figure(1)
plt.plot(alpha, PSI_th, color = 'cornflowerblue', linestyle=(0, (5, 5)), label = r'$\Psi$ (equilibrium)')
plt.plot(alpha, PSI_0, color = 'black', linestyle=(0, (5, 5)), label = r'$\Psi$ (one-shot)')
plt.xlabel(r'$\alpha$', fontsize=16)
plt.xticks(alpha)
plt.legend(bbox_to_anchor =(0.5,-0.3), loc='lower center', ncol = 2, fontsize=12)
fig.savefig(plots_path + 'th_fairness_metric.png', dpi = 1000, bbox_inches='tight')
plt.show()