import pandas as pd
import math
import os
import datetime as dt
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


unit_ = 'US'
ifile = 'RMSE tr model 148 pars final.csv'  
bins=range(0,30,1) # steady state model v1
ofile = 'fig_hist_rmse_148par_tr_model_final.png'
print(f'Reading file {ifile}')

data = pd.read_csv(ifile)
fig, ax = plt.subplots()
fig.set_size_inches(1.8, 1.5) 

n_bins = 10
#
#pval_cur = range(nsamples)  # 1000 realizations
fitness = data.RMSE
if unit_=='US':
    fitness = data.RMSE*3.28084
else:
    fitness = data.RMSE

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

kwargs = dict(ecolor='k', color='c', capsize=2,
        elinewidth=1, linewidth=0.6, ms=5, alpha=0.95)
plt.rc('grid', linestyle="dotted", color='gray', alpha = 0.15)

plot_option = 2
if plot_option == 0:  # Point plots
    plt.scatter(pval_cur, fitness, s=100, cmap='rainbow',
                marker='o', linewidths=0.25, edgecolors='k', alpha=0.75)

    plt.xlabel('Realizations')
    # plt.ylabel('Outflow southwest (m3/day)')
    plt.ylabel('RMSE (m)')
    plt.grid(True)
    ofile = 'fig_scatter_par_base_0_rmse.png'
elif plot_option == 1: # Hist plot
    counts, bins ,patches = plt.hist(fitness, bins, density=False,
        edgecolor='black', linewidth=0.5, alpha = 0.5)
    plt.ylabel('Frequency')
    plt.xlabel('RMSE (m)')
    ax.grid(True)
    #plt.show()

elif plot_option == 2:
    # Density Plot and Histogram 

    sns.distplot(fitness, hist=True, kde=False, 
                bins=bins, color = 'darkblue', 
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 1})
#        sns.distplot(fitness, bins=n_bins,
#             hist_kws=dict(cumulative=True),
#             kde_kws=dict(cumulative=True))                    

#        sns.distplot(fitness, hist = False, kde = True,
#                        kde_kws = {'shade': True, 'linewidth': 2}, 
#                        label = airline)
    #plt.ylabel('Density')
    plt.ylabel('Frequency')
    plt.xlabel('RMSE (ft)')

    ofile = 'fig_dens_par_base_0_ss_rmse.png'

    ax.grid(True)
    fig.savefig(ofile, dpi=200, transparent=True, bbox_inches='tight')

    plt.show()