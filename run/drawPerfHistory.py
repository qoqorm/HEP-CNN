#!/usr/bin/env python
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import sys, os

metric = 'CPU'

for d in sys.argv[1:]:
    hostInfo, pars = d.split('/',1)
    hostAlias, hostSpec = hostInfo[5:].split('_',1)

    nthreads = int(pars.split('/')[0].split('_')[-1])
    nbatch   = int(pars.split('/')[1].split('_')[-1])

    usage = pd.read_csv('%s/batchHistory.csv' % d)
    usage2 = pd.read_csv('%s/usageHistory.csv' % d)

    usage.append(usage2, ignore_index=True)
    usage['Datetime'] = pd.to_datetime(usage['Datetime'], format='%Y-%m-%d %H-%M-%S')
    beginTime = min(usage['Datetime'])

    usage['time'] = (usage['Datetime']-beginTime).dt.total_seconds()

    plt.plot(usage['time'], usage[metric])

plt.show()
