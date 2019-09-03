#!/usr/bin/env python
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
import argparse

metrics_all = ['CPU', 'RSS', 'VMSize', 'Read', 'Write']
metrics_opts = {'CPU':('%', 1), 'RSS':('GB',1e9), 'VMSize':('GB',1e9), 'Read':('MB',1e6), 'Write':('MB',1e6)}

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--metrics', type=str, nargs='*', action='append', default=[['CPU']], choices=metrics_all+["all"],
                    help="List of metrics (CPU,RSS,VMSize,Read,Write,all)")
parser.add_argument('-d', '--dirs', type=str, nargs='+', action='append', default=[],
                    help="Directories with log messages")
args = parser.parse_args()

metrics = []
for x in args.metrics: metrics.extend(x)
metrics = list(set(metrics))
if 'all' in metrics: metrics = metrics_all

dirs = []
for dd in args.dirs:
    for d in dd:
        if not os.path.exists(d+'/batchHistory_0.csv'): continue
        dirs.append(d)

cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 5

plt.rcParams['figure.figsize'] = (7, len(metrics)*2)
maxTime = 0
if len(metrics) > 1:
    for d in dirs:
        for i, metric in enumerate(metrics):
            hostInfo, pars = d.split('/',1)
            hostAlias, hostSpec = hostInfo.replace('perf_', '').split('_',1)

            ax = plt.subplot(len(metrics), 1, i+1)

            for ii in range(64):
                if not os.path.exists('%s/batchHistory_%d.csv' % (d, ii)): continue
                usage = pd.read_csv('%s/batchHistory_%d.csv' % (d, ii))
                usage2 = pd.read_csv('%s/usageHistory_%d.csv' % (d, ii))
                usage = usage.append(usage2, ignore_index=True)

                usage['Datetime'] = pd.to_datetime(usage['Datetime'], format='%Y-%m-%d %H-%M-%S')
                usage = usage.sort_values(['Datetime'])
                beginTime = min(usage['Datetime'])

                usage['time'] = (usage['Datetime']-beginTime).dt.total_seconds()
                maxTime = max(max(usage['time']), maxTime)
                unit, scale = metrics_opts[metric]

                plt.plot(usage['time'], usage[metric]/scale, '.-', label=('rank%d'%ii))#, c=cols[i], label=(pars.replace('__', ' ')))
            plt.grid(linestyle=':')
            if i == len(metrics)-1: plt.xlabel('time')
            plt.ylabel('%s(%s)' % (metric, unit))
            #ax.set_xlim([0, 1000])
            #plt.yscale('log')
            #if metric == 'CPU':
            #    ax.set_ylim([0, 6000])

            plt.legend()
        plt.tight_layout()

        plt.savefig('%s/%s.png' % (d, metric))
        plt.show()

if len(dirs) > 1:
    plt.rcParams['figure.figsize'] = (7, len(dirs)*2)
    for metric in metrics:
        for i, d in enumerate(dirs):
            hostInfo, pars = d.split('/',1)
            hostAlias, hostSpec = hostInfo.replace('perf_', '').split('_',1)

            ax = plt.subplot(len(dirs), 1, i+1)
            plt.title(pars.replace('__', ' '))

            for ii in range(64):
                if not os.path.exists('%s/batchHistory_%d.csv' % (d, ii)): continue
                usage = pd.read_csv('%s/batchHistory_%d.csv' % (d, ii))
                usage2 = pd.read_csv('%s/usageHistory_%d.csv' % (d, ii))
                usage = usage.append(usage2, ignore_index=True)

                usage['Datetime'] = pd.to_datetime(usage['Datetime'], format='%Y-%m-%d %H-%M-%S')
                usage = usage.sort_values(['Datetime'])
                beginTime = min(usage['Datetime'])

                usage['time'] = (usage['Datetime']-beginTime).dt.total_seconds()
                unit, scale = metrics_opts[metric]

                #plt.plot(usage['time'], usage[metric]/scale, '.-', c=cols[i], label=(pars.replace('__', ' ')))
                plt.plot(usage['time'], usage[metric]/scale, '.-', label=('rank%d'%ii))
            plt.grid(linestyle=':')
            if i == len(dirs)-1: plt.xlabel('time')
            plt.ylabel('%s(%s)' % (metric, unit))
            if maxTime > 0: ax.set_xlim([0, maxTime])
            #plt.yscale('log')
            #if metric == 'CPU':
            #    ax.set_ylim([0, 6000])

            plt.legend()
        plt.tight_layout()

        plt.savefig('%s.png' % (metric))
        plt.show()
