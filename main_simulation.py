from simulation import simulate
from estimate import *
from config import *
from numerical import *
from scipy.optimize import fsolve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns


'''
  Simulate
'''
simulation_output = simulate()

rho = simulation_output['rho']
rk = simulation_output['rk']
pos = simulation_output['pos']
user_id = simulation_output['user_id']
ranking = simulation_output['ranking']
epsilon = simulation_output['epsilon']
B = simulation_output['B']

'''
  Estimate
'''
print('- Estimate -')
estimation_output = estimate(rho, ranking, B)

R = estimation_output['R']
RP = estimation_output['RP']
b_emp = estimation_output['b_emp']
b_emp_std = estimation_output['b_emp_std']
BP = estimation_output['BP']
BP_std = estimation_output['BP_std']
psi = estimation_output['psi']
psi_std = estimation_output['psi_std']
PSI = estimation_output['PSI']
PSI_std = estimation_output['PSI_std']

b_bar = np.array([np.sum(b_emp[s]) for s in range(S)]) / M
b_bar_std = np.std(np.array([np.sum(b_emp[s]) for s in range(S)]) / M)

q_th = np.array([fsolve(implicit_equations, 0.5*np.ones(V), args = (mu, theta, rho[s], gamma, rk[s], pos[s], U, V)) for s in range(S)], dtype = np.double)
b_th = np.multiply(mu, (1-q_th))
psi_th = np.array([compute_psi(R[s], b_th[s], V) for s in range(S)], dtype = np.double)
PSI_th = np.array([compute_PSI(psi_th[s]) for s in range(S)], dtype = np.double)

'''
  Plot crowding heatmaps
'''
print('- Plot crowding -')
widths = [1 for s in range(3)]
heights = [1, 0.05, 0.2, 1, 0.05]
fig, ax = plt.subplots(5, 3, figsize=(16,12), gridspec_kw = dict(width_ratios=widths, height_ratios=heights))
fig.add_subplot(111, frameon = False)
for s in range(3):
  hm = sns.heatmap(pd.DataFrame(rho[s]), cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True), cbar=False, vmin = 0, vmax = rho.max(), xticklabels = False, yticklabels = False,  ax=ax[0,s])
  hm.set_title(labels[s], fontsize=14, pad=5)
  ax[0,0].set_ylabel(r'$\rho$', fontsize = 16, rotation = 0, verticalalignment='center', labelpad=20)
  ax[1,s].imshow(np.expand_dims(R[s], axis = 0), cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True), vmin = 0, vmax = rho.max(), aspect='auto')
  ax[1,s].set_xticks([])
  ax[1,s].set_yticks([])
  ax[1,0].set_ylabel(r'$\mathcal{R}$', fontsize = 16, rotation = 0,verticalalignment='center', labelpad=20)
  ax[2,s].axis('off')
for s in range(3,6):
  hm = sns.heatmap(pd.DataFrame(rho[s]), cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True), cbar=False, vmin = 0, vmax = rho.max(), xticklabels = False, yticklabels = False,  ax=ax[3,s-3])
  hm.set_title(labels[s], fontsize=14, pad=5)
  ax[3,0].set_ylabel(r'$\rho$', fontsize = 16, rotation = 0, verticalalignment='center', labelpad=20)
  ax[4,s-3].imshow(np.expand_dims(R[s], axis = 0), cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True), vmin = 0, vmax = rho.max(), aspect='auto')
  ax[4,s-3].set_xticks([])
  ax[4,s-3].set_yticks([])
  ax[4,0].set_ylabel(r'$\mathcal{R}$', fontsize = 16, rotation = 0, verticalalignment='center', labelpad=20)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.tight_layout()
plt.savefig(plots_path + 'heatmap.png', dpi = 1000)
plt.show()

'''
  Plot relevance barplot
'''
print('- Plot relevance -')
# Barplot booking/relevance ordered by relevance
fig, axs = plt.subplots(2, 3, figsize=(18,12), sharey = True)
fig.add_subplot(111, frameon = False)
vs = [v for v in range(V)]
for s in range(3):
  i = 0
  items = np.argsort(RP[s])
  BP_th = b_th[s]/np.sum(b_th[s])
  axs[0,s].plot(BP_th[items], color = 'darkslategray', linestyle='dashed', linewidth = 1)
  for v in items:
    yerr = z * BP_std[s,v] / np.sqrt(M)
    axs[0,s].bar(i, RP[s,v], color = 'bisque')
    axs[0,s].bar(i, BP[s,v], color = 'lightsteelblue', alpha = 0.5)
    axs[0,s].set_title(labels[s], fontsize = 14, pad = 2)
    axs[0,s].errorbar(i, BP[s,v], yerr = yerr, color = 'cornflowerblue', capsize = 3)
    i += 1
    axs[0,s].set_xticks([])
for s in range(3,6):
  i = 0
  items = np.argsort(RP[s])
  BP_th = b_th[s]/np.sum(b_th[s])
  axs[1,s-3].plot(BP_th[items], color = 'darkslategray', linestyle='dashed', linewidth = 1)
  for v in items:
    yerr = z * BP_std[s,v] / np.sqrt(M)
    axs[1,s-3].bar(i, RP[s,v], color = 'bisque')
    axs[1,s-3].bar(i, BP[s,v], color = 'lightsteelblue', alpha = 0.5)
    axs[1,s-3].set_title(labels[s], fontsize = 14, pad = 2)
    axs[1,s-3].errorbar(i, BP[s,v], yerr = yerr, color = 'cornflowerblue', capsize = 3)
    i += 1
    axs[1,s-3].set_xticks([])
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.subplots_adjust(wspace=0, hspace=0)
relevance_1_patch = mpatches.Patch(color='bisque', label=r'$\mathcal{R}$')
booking_patch = mpatches.Patch(color='lightsteelblue', label=r'$\bar{b}$ (simulated)')
theoretical_line = Line2D([0], [0], linestyle='dashed', color='darkslategray', label=r'$\bar{b}$ (theoretical)')
plt.legend(handles=[relevance_1_patch, booking_patch, theoretical_line], bbox_to_anchor =(0.5,-0.1), loc='lower center', fontsize=16, ncol=3)
plt.tight_layout()
plt.savefig(plots_path + 'relevance.png', dpi = 1000)
plt.show()

'''
  Plot fairness metric
'''
print('- Plot fairness -')
yerr = z * PSI_std / np.sqrt(M)
fig = plt.figure(1)
plt.plot(alpha, PSI, color = 'lightsteelblue', linestyle='solid', label = r'$\Psi$ (simulated)')
plt.plot(alpha, PSI_th, color = 'darkslategray', linestyle=(0, (5, 5)), label = r'$\Psi$ (theoretical)')
plt.errorbar(alpha, PSI, yerr = yerr, color = 'cornflowerblue', capsize = 3)
plt.xlabel(r'$\alpha$', fontsize=16)
plt.xticks(alpha)
plt.legend(bbox_to_anchor =(0.5,-0.3), loc='lower center', ncol = 2, fontsize=12)
fig.savefig(plots_path + 'fairness_metric.png', dpi = 1000, bbox_inches='tight')
plt.show()