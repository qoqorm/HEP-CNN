#!/usr/bin/env python
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import sys, os

metric = 'CPU'

dirs = [d for d in sys.argv[1:] if os.path.exists(d+'/batchHistory.csv')]
cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

plt.rcParams['figure.figsize'] = (7, len(dirs)*1.2)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 5

for i, d in enumerate(dirs):
    hostInfo, pars = d.split('/',1)
    hostAlias, hostSpec = hostInfo.replace('perf_', '').split('_',1)

    usage = pd.read_csv('%s/batchHistory.csv' % d)
    usage2 = pd.read_csv('%s/usageHistory.csv' % d)

    usage.append(usage2, ignore_index=True)
    usage['Datetime'] = pd.to_datetime(usage['Datetime'], format='%Y-%m-%d %H-%M-%S')
    beginTime = min(usage['Datetime'])

    usage['time'] = (usage['Datetime']-beginTime).dt.total_seconds()

    ax = plt.subplot(len(dirs), 1, i+1)
    plt.plot(usage['time'], usage[metric], '.-', c=cols[i], label=(pars.replace('__', ' ')))
    plt.grid(linestyle=':')
    if i == len(dirs)-1: plt.xlabel('time')
    plt.ylabel(metric)
    ax.set_xlim([0, 1000])
    #plt.yscale('log')
    if metric == 'CPU':
        ax.set_ylim([0, 6000])

    plt.legend()
plt.tight_layout()

plt.savefig('%s.png' % (metric))
plt.show()
