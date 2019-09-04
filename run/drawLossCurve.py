#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd
import sys, os

plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 5

plt.rcParams['figure.figsize'] = (7, 3*2)
ax1 = plt.subplot(2, 2, 1, yscale='log', ylabel='Loss(train)')
ax2 = plt.subplot(2, 2, 3, yscale='log', ylabel='Loss(val)')
ax3 = plt.subplot(2, 2, 2, ylabel='Accuracy(train)')
ax4 = plt.subplot(2, 2, 4, ylabel='Accuracy(val)', xlabel='epoch')
for d in sys.argv[1:]:
    df = pd.read_csv(d)

    label = d.split('/')[-2]

    ax1.plot(df['loss'], '.-', label=label)
    ax2.plot(df['val_loss'], '.-', label=label)

    ax3.plot(df['acc'], '.-', label=label)
    ax4.plot(df['val_acc'], '.-', label=label)
    #ax.set_ylim([0,1])

plt.legend()

plt.tight_layout()
plt.show()
