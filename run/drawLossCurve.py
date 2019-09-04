#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd
import sys, os

plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 5

plt.rcParams['figure.figsize'] = (7, 2*2)
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
for d in sys.argv[1:]:
    df = pd.read_csv(d)

    ax1.plot(df['loss'], '.-', label='train')
    ax1.plot(df['val_loss'], '.-', label='val')
    #ax1.ylabel('Loss')
    #ax1.yscale('log')

    ax2.plot(df['acc'], '.-', label='train')
    ax2.plot(df['val_acc'], '.-', label='val')
    #ax.set_ylim([0,1])
    #ax2.ylabel('Accuracy')
    #ax2.xlabel('epoch')

plt.tight_layout()
plt.show()
